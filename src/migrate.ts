import 'dotenv/config'
import OpenAI from 'openai'
import { createRailwayClient, createZillizClient, DataType } from './milvus.js'

// ── 配置 ────────────────────────────────────────────────────────────────────
const SOURCE_COLLECTION = 'easykol_milvus_beta'
const TARGET_COLLECTION = 'easykol_influencer_v1'
const PLATFORM_FILTER = "platform == 'YOUTUBE'"
const QUERY_BATCH_SIZE = 100  // 缩小批次，减少单次连接时间
const INSERT_BATCH_SIZE = 50
const AI_CONCURRENCY = 5
const MIGRATE_LIMIT = 1000    // 最多迁移条数，设为 Infinity 则不限

const ai = new OpenAI({
  apiKey: process.env.AIHUBMIX_API_KEY ?? '',
  baseURL: 'https://aihubmix.com/v1',
})
const MODEL = process.env.AIHUBMIX_MODEL ?? 'gemini-2.5-flash'

// ── AI 摘要生成 ──────────────────────────────────────────────────────────────
async function generateSummary(videoTexts: string): Promise<string> {
  const prompt = `Act as a professional media content analyst. Please analyze the provided list of video titles and summarize the blogger's niche and content.

### Constraints:
1. Language: You must respond in the SAME language as the provided input list.
2. Format: Provide a single, concise sentence.
3. Content: Accurately summarize the blogger's "niche/field" and "core content themes."
4. Tone: Professional and direct.

### Input Data:
${videoTexts}`

  const res = await ai.chat.completions.create({
    model: MODEL,
    messages: [{ role: 'user', content: prompt }],
    max_tokens: 256,
  })
  return res.choices[0]?.message?.content?.trim() ?? ''
}

// ── 并发控制 ─────────────────────────────────────────────────────────────────
async function mapWithConcurrency<T, R>(
  items: T[],
  fn: (item: T, idx: number) => Promise<R>,
  concurrency: number,
): Promise<R[]> {
  const results: R[] = new Array(items.length)
  let cursor = 0
  async function worker() {
    while (cursor < items.length) {
      const idx = cursor++
      results[idx] = await fn(items[idx], idx)
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, worker))
  return results
}

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

// ── 自动创建目标集合（若不存在）────────────────────────────────────────────────
async function ensureTargetCollection(vectorDim: number) {
  const railway = createRailwayClient()
  const exists = await railway.hasCollection({ collection_name: TARGET_COLLECTION })
  if (exists.value) return
  console.log(`  目标集合不存在，正在创建 [${TARGET_COLLECTION}]，向量维度=${vectorDim}...`)
  await railway.createCollection({
    collection_name: TARGET_COLLECTION,
    fields: [
      { name: 'id',                data_type: DataType.VarChar,     is_primary_key: true, max_length: 128 },
      { name: 'userId',            data_type: DataType.VarChar,     max_length: 128 },
      { name: 'account',           data_type: DataType.VarChar,     max_length: 255 },
      { name: 'platform',          data_type: DataType.VarChar,     max_length: 64  },
      { name: 'nickname',          data_type: DataType.VarChar,     max_length: 255 },
      { name: 'followerCount',     data_type: DataType.Int64 },
      { name: 'averagePlayCount',  data_type: DataType.Double },
      { name: 'lastPublishedTime', data_type: DataType.Int64 },
      { name: 'region',            data_type: DataType.VarChar,     max_length: 64  },
      { name: 'signature',         data_type: DataType.VarChar,     max_length: 2000 },
      { name: 'videoTexts',        data_type: DataType.VarChar,     max_length: 65535 },
      { name: 'email',             data_type: DataType.VarChar,     max_length: 255 },
      { name: 'dense_vector',      data_type: DataType.FloatVector, dim: vectorDim },
      { name: 'meta',              data_type: DataType.JSON },
      { name: 'reserved1',         data_type: DataType.VarChar,     max_length: 1000 },
      { name: 'reserved2',         data_type: DataType.VarChar,     max_length: 1000 },
      { name: 'reserved3',         data_type: DataType.VarChar,     max_length: 1000 },
      { name: 'createdAt',         data_type: DataType.Int64 },
      { name: 'updatedAt',         data_type: DataType.Int64 },
      { name: 'language',          data_type: DataType.VarChar,     max_length: 64  },
      { name: 'ai_summary',        data_type: DataType.VarChar,     max_length: 2000 },
    ],
  })
  await railway.createIndex({
    collection_name: TARGET_COLLECTION,
    field_name: 'dense_vector',
    index_name: 'dense_vector_idx',
    extra_params: { index_type: 'HNSW', metric_type: 'COSINE', params: JSON.stringify({ M: 16, efConstruction: 200 }) },
  })
  console.log(`  ✓ 集合已创建并建立索引`)
}

// ── 获取 Railway 已有的所有 id（断点续传用）────────────────────────────────────
async function fetchExistingIds(): Promise<Set<string>> {
  const railway = createRailwayClient()
  console.log('  连接 Railway Milvus...')
  try {
    await railway.loadCollection({ collection_name: TARGET_COLLECTION })
  } catch (err: any) {
    if (err.message?.includes('CollectionNotExists') || err.message?.includes('collection not found')) {
      console.log('  目标集合尚不存在，将从头开始迁移')
      return new Set<string>()
    }
    throw err
  }
  console.log('  集合已加载，开始读取已有 id...')
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
  return existing
}

// ── 主流程 ───────────────────────────────────────────────────────────────────
async function main() {
  const zilliz = createZillizClient()

  // 获取源集合所有字段名 & 向量维度
  const desc = await withRetry(() => zilliz.describeCollection({ collection_name: SOURCE_COLLECTION }))
  const allFields = desc.schema.fields.map((f) => f.name)
  const vectorField = desc.schema.fields.find((f) => f.data_type === 'FloatVector' || (f as any).dataType === 101)
  const vectorDim = Number((vectorField as any)?.type_params?.find((p: any) => p.key === 'dim')?.value ?? 1536)

  // 确保目标集合存在
  await ensureTargetCollection(vectorDim)

  // 断点续传：获取已插入的 id
  console.log('检查断点，读取 Railway 已有 id...')
  const existingIds = await fetchExistingIds()
  console.log(`Railway 已有 ${existingIds.size} 条，将跳过重复\n`)

  console.log(`开始迁移 Zilliz[${SOURCE_COLLECTION}] → Railway[${TARGET_COLLECTION}]`)
  console.log(`过滤条件: ${PLATFORM_FILTER}\n`)

  let offset = 0
  let totalFetched = 0
  let totalInserted = 0
  let totalSkippedDup = 0

  while (true) {
    if (totalInserted >= MIGRATE_LIMIT) break

    // 1. 拉取数据
    const queryRes = await withRetry(() => zilliz.query({
      collection_name: SOURCE_COLLECTION,
      filter: PLATFORM_FILTER,
      output_fields: allFields,
      limit: QUERY_BATCH_SIZE,
      offset,
    }))

    const rows = queryRes.data
    if (!rows?.length) break

    totalFetched += rows.length

    // 过滤：跳过已存在 + videoTexts 为空
    let newRows = rows.filter((r) => {
      if (existingIds.has(String(r.id))) return false
      if (!String(r.videoTexts ?? '').trim()) return false
      return true
    })
    // 截断到剩余配额
    const remaining = MIGRATE_LIMIT - totalInserted
    if (newRows.length > remaining) newRows = newRows.slice(0, remaining)
    totalSkippedDup += rows.length - newRows.length

    console.log(`[${offset + 1}~${offset + rows.length}] 拉取 ${rows.length} 条，新增 ${newRows.length} 条，生成 ai_summary...`)

    if (newRows.length > 0) {
      // 2. 并发生成 ai_summary
      const summaries = await mapWithConcurrency(
        newRows,
        async (row, i) => {
          const idx = offset + rows.indexOf(row) + 1
          const nick = String(row.nickname ?? '').slice(0, 20).padEnd(20)
          try {
            const summary = await generateSummary(String(row.videoTexts))
            console.log(`  [${String(idx).padStart(5)}] ${nick}`)
            console.log(`           ${summary || '[WARN: AI 返回空]'}`)
            return summary
          } catch (err: any) {
            console.log(`  [${String(idx).padStart(5)}] ${nick} → [ERROR: ${String(err?.message ?? '').slice(0, 60)}]`)
            return ''
          }
        },
        AI_CONCURRENCY,
      )

      // 3. 组装数据
      const records = newRows.map((row, i) => ({
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
        ai_summary:        summaries[i],
      }))

      // 4. 每批重新建连接，避免长连接被 proxy 切断
      const railway = createRailwayClient()
      await railway.loadCollection({ collection_name: TARGET_COLLECTION })
      for (const batch of chunk(records, INSERT_BATCH_SIZE)) {
        await railway.insert({ collection_name: TARGET_COLLECTION, data: batch })
        totalInserted += batch.length
      }

      await railway.flushSync({ collection_names: [TARGET_COLLECTION] })
      const stats = await railway.getCollectionStatistics({ collection_name: TARGET_COLLECTION })
      const flushedCount = stats.stats.find((s) => s.key === 'row_count')?.value ?? '?'
      console.log(`  ✓ 本批写入 ${records.length} 条，📊 Railway 累计已落盘: ${flushedCount} 条\n`)

      // 更新已知 id 缓存
      records.forEach((r) => existingIds.add(r.id))
    }

    if (rows.length < QUERY_BATCH_SIZE) break
    offset += rows.length
  }

  console.log('─────────────────────────────────')
  console.log(`迁移完成！Zilliz 拉取: ${totalFetched} 条`)
  console.log(`跳过（重复/空）: ${totalSkippedDup} 条`)
  console.log(`新写入: ${totalInserted} 条`)
}

main().catch((err) => {
  console.error('迁移失败:', err.message ?? err)
  process.exit(1)
})
