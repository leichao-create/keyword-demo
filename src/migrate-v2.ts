import 'dotenv/config'
import { createRailwayClient, DataType } from './milvus.js'
import { FunctionType } from '@zilliz/milvus2-sdk-node'

// ── 配置 ────────────────────────────────────────────────────────────────────
const SOURCE_COLLECTION = 'easykol_influencer_v1'
const TARGET_COLLECTION = 'easykol_influencer_v2'
const QUERY_BATCH_SIZE = 200
const INSERT_BATCH_SIZE = 100
const MIGRATE_LIMIT = Infinity  // 全量迁移

// 所有需要从 v1 读取的字段（不含 dense_vector，单独处理）
const SOURCE_OUTPUT_FIELDS = [
  'id', 'userId', 'account', 'platform', 'nickname',
  'followerCount', 'averagePlayCount', 'lastPublishedTime',
  'region', 'signature', 'videoTexts', 'email',
  'dense_vector', 'meta',
  'reserved1', 'reserved2', 'reserved3',
  'createdAt', 'updatedAt', 'language', 'ai_summary',
]

function chunk<T>(arr: T[], size: number): T[][] {
  const out: T[][] = []
  for (let i = 0; i < arr.length; i += size) out.push(arr.slice(i, i + size))
  return out
}

async function withRetry<T>(fn: () => Promise<T>, retries = 3, delayMs = 5000): Promise<T> {
  for (let i = 0; i < retries; i++) {
    try {
      return await fn()
    } catch (err: any) {
      const isRetryable = err.message?.includes('DEADLINE_EXCEEDED') || err.message?.includes('UNAVAILABLE')
      if (!isRetryable || i === retries - 1) throw err
      console.log(`  [重试 ${i + 1}/${retries}] ${err.message?.slice(0, 60)}，${delayMs / 1000}s 后重试...`)
      await new Promise((r) => setTimeout(r, delayMs))
    }
  }
  throw new Error('unreachable')
}

// ── 创建 v2 集合（带 BM25 全文检索）────────────────────────────────────────
async function ensureV2Collection(vectorDim: number) {
  const railway = createRailwayClient()
  const exists = await railway.hasCollection({ collection_name: TARGET_COLLECTION })
  if (exists.value) {
    console.log(`  目标集合 [${TARGET_COLLECTION}] 已存在，跳过创建`)
    return
  }

  console.log(`  创建集合 [${TARGET_COLLECTION}]，向量维度=${vectorDim}，启用 BM25...`)

  await railway.createCollection({
    collection_name: TARGET_COLLECTION,
    // 启用动态字段，方便后续扩展
    enable_dynamic_field: false,
    fields: [
      { name: 'id',                data_type: DataType.VarChar,          is_primary_key: true, max_length: 128 },
      { name: 'userId',            data_type: DataType.VarChar,          max_length: 128 },
      { name: 'account',           data_type: DataType.VarChar,          max_length: 255 },
      { name: 'platform',          data_type: DataType.VarChar,          max_length: 64  },
      { name: 'nickname',          data_type: DataType.VarChar,          max_length: 255 },
      { name: 'followerCount',     data_type: DataType.Int64 },
      { name: 'averagePlayCount',  data_type: DataType.Double },
      { name: 'lastPublishedTime', data_type: DataType.Int64 },
      { name: 'region',            data_type: DataType.VarChar,          max_length: 64  },
      { name: 'signature',         data_type: DataType.VarChar,          max_length: 2000 },
      // BM25 源字段：enable_analyzer 让 Milvus 自动 tokenize
      {
        name: 'videoTexts',
        data_type: DataType.VarChar,
        max_length: 65535,
        enable_analyzer: true,
        analyzer_params: { type: 'english' },  // 英文分词器
      },
      { name: 'email',             data_type: DataType.VarChar,          max_length: 255 },
      { name: 'dense_vector',      data_type: DataType.FloatVector,      dim: vectorDim },
      // BM25 输出字段：存储稀疏向量，由 Function 自动填充，无需手动插入
      { name: 'sparse_vector',     data_type: DataType.SparseFloatVector },
      { name: 'meta',              data_type: DataType.JSON },
      { name: 'reserved1',         data_type: DataType.VarChar,          max_length: 1000 },
      { name: 'reserved2',         data_type: DataType.VarChar,          max_length: 1000 },
      { name: 'reserved3',         data_type: DataType.VarChar,          max_length: 1000 },
      { name: 'createdAt',         data_type: DataType.Int64 },
      { name: 'updatedAt',         data_type: DataType.Int64 },
      { name: 'language',          data_type: DataType.VarChar,          max_length: 64  },
      {
        name: 'ai_summary',
        data_type: DataType.VarChar,
        max_length: 2000,
        enable_analyzer: true,
        analyzer_params: { type: 'english' },
      },
    ],
    // BM25 Function：Milvus 自动将 videoTexts → sparse_vector
    functions: [
      {
        name: 'bm25_video_texts',
        description: 'BM25 full-text search on videoTexts',
        type: FunctionType.BM25,
        input_field_names: ['videoTexts'],
        output_field_names: ['sparse_vector'],
      },
    ],
  })

  // 稠密向量索引
  await railway.createIndex({
    collection_name: TARGET_COLLECTION,
    field_name: 'dense_vector',
    index_name: 'dense_idx',
    extra_params: {
      index_type: 'HNSW',
      metric_type: 'COSINE',
      params: JSON.stringify({ M: 16, efConstruction: 200 }),
    },
  })

  // 稀疏向量索引（BM25）
  await railway.createIndex({
    collection_name: TARGET_COLLECTION,
    field_name: 'sparse_vector',
    index_name: 'sparse_idx',
    extra_params: {
      index_type: 'SPARSE_INVERTED_INDEX',
      metric_type: 'BM25',
      params: JSON.stringify({ bm25_k1: 1.2, bm25_b: 0.75 }),
    },
  })

  console.log(`  ✓ 集合已创建，dense + sparse 索引已建立`)
}

// ── 读取 v2 已有 id（断点续传）──────────────────────────────────────────────
async function fetchExistingIds(): Promise<Set<string>> {
  const railway = createRailwayClient()
  try {
    await railway.loadCollection({ collection_name: TARGET_COLLECTION })
  } catch (err: any) {
    if (err.message?.includes('CollectionNotExists') || err.message?.includes('collection not found')) {
      return new Set<string>()
    }
    throw err
  }
  const existing = new Set<string>()
  let page = 0
  const pageSize = 1000
  while (true) {
    const res = await railway.query({
      collection_name: TARGET_COLLECTION,
      output_fields: ['id'],
      limit: pageSize,
      offset: page * pageSize,
    })
    if (!res.data?.length) break
    res.data.forEach((r) => existing.add(String(r.id)))
    if (res.data.length < pageSize) break
    page++
  }
  console.log(`  v2 已有 ${existing.size} 条记录`)
  return existing
}

// ── 主流程 ───────────────────────────────────────────────────────────────────
async function main() {
  const source = createRailwayClient()

  // 读取 v1 schema，获取向量维度
  const desc = await source.describeCollection({ collection_name: SOURCE_COLLECTION })
  const vectorField = desc.schema.fields.find((f: any) => f.data_type === 'FloatVector' || f.data_type === 101)
  const vectorDim = Number((vectorField as any)?.type_params?.find((p: any) => p.key === 'dim')?.value ?? 3072)
  console.log(`源集合 [${SOURCE_COLLECTION}]，向量维度: ${vectorDim}`)

  // 确保 v2 集合存在
  await ensureV2Collection(vectorDim)

  // 断点续传
  console.log('\n检查断点...')
  const existingIds = await fetchExistingIds()

  // 加载 v1
  await withRetry(() => source.loadCollection({ collection_name: SOURCE_COLLECTION }))

  const target = createRailwayClient()
  await target.loadCollection({ collection_name: TARGET_COLLECTION })

  let offset = 0
  let totalFetched = 0
  let totalInserted = 0
  let totalSkipped = 0

  console.log(`\n开始迁移 v1 → v2（全量，跳过已存在）\n`)

  while (true) {
    if (totalInserted >= MIGRATE_LIMIT) break

    // 1. 分页读取 v1
    const queryRes = await withRetry(() =>
      source.query({
        collection_name: SOURCE_COLLECTION,
        output_fields: SOURCE_OUTPUT_FIELDS,
        limit: QUERY_BATCH_SIZE,
        offset,
      }),
    )

    const rows = queryRes.data
    if (!rows?.length) break
    totalFetched += rows.length

    // 2. 过滤已存在 + videoTexts 为空
    let newRows = rows.filter((r) => {
      if (existingIds.has(String(r.id))) return false
      if (!String(r.videoTexts ?? '').trim()) return false
      return true
    })
    const remaining = MIGRATE_LIMIT - totalInserted
    if (newRows.length > remaining) newRows = newRows.slice(0, remaining)
    totalSkipped += rows.length - newRows.length

    console.log(`[${offset + 1}~${offset + rows.length}] 拉取 ${rows.length}，新增 ${newRows.length}，跳过 ${rows.length - newRows.length}`)

    if (newRows.length > 0) {
      // 3. 组装记录（不需要传 sparse_vector，Milvus BM25 Function 自动生成）
      const records = newRows.map((row) => ({
        id:                String(row.id                ?? ''),
        userId:            String(row.userId             ?? ''),
        account:           String(row.account            ?? ''),
        platform:          String(row.platform           ?? ''),
        nickname:          String(row.nickname           ?? ''),
        followerCount:     Number(row.followerCount      ?? 0),
        averagePlayCount:  Number(row.averagePlayCount   ?? 0),
        lastPublishedTime: Number(row.lastPublishedTime  ?? 0),
        region:            String(row.region             ?? ''),
        signature:         String(row.signature          ?? ''),
        videoTexts:        String(row.videoTexts         ?? ''),
        email:             String(row.email              ?? ''),
        dense_vector:      row.dense_vector              as number[],
        meta:              row.meta                      ?? {},
        reserved1:         String(row.reserved1          ?? ''),
        reserved2:         String(row.reserved2          ?? ''),
        reserved3:         String(row.reserved3          ?? ''),
        createdAt:         Number(row.createdAt          ?? 0),
        updatedAt:         Number(row.updatedAt          ?? 0),
        language:          String(row.language           ?? ''),
        ai_summary:        String(row.ai_summary         ?? ''),
      }))

      // 4. 分批写入
      for (const batch of chunk(records, INSERT_BATCH_SIZE)) {
        await withRetry(() => target.insert({ collection_name: TARGET_COLLECTION, data: batch }))
        totalInserted += batch.length
        existingIds.add(...batch.map((r) => r.id))
      }

      await target.flushSync({ collection_names: [TARGET_COLLECTION] })
      const stats = await target.getCollectionStatistics({ collection_name: TARGET_COLLECTION })
      const rowCount = stats.stats.find((s: any) => s.key === 'row_count')?.value ?? '?'
      console.log(`  ✓ 写入 ${records.length} 条，v2 累计落盘: ${rowCount} 条\n`)
    }

    if (rows.length < QUERY_BATCH_SIZE) break
    offset += rows.length
  }

  console.log('─────────────────────────────────')
  console.log(`迁移完成！v1 拉取: ${totalFetched} 条`)
  console.log(`跳过（重复/空）: ${totalSkipped} 条`)
  console.log(`新写入 v2: ${totalInserted} 条`)
}

main().catch((err) => {
  console.error('迁移失败:', err.message ?? err)
  process.exit(1)
})
