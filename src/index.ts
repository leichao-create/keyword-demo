import 'dotenv/config'
import { createRailwayClient, createZillizClient } from './milvus.js'
import { DataType, FunctionType } from '@zilliz/milvus2-sdk-node'

const COLLECTION_NAME = 'easykol_influencer_v1'
const SOURCE_COLLECTION = 'easykol_milvus_beta'

async function main() {
  const zilliz = createZillizClient()
  const railway = createRailwayClient()

  // 从 Zilliz 获取 dense_vector 维度
  const desc = await zilliz.describeCollection({ collection_name: SOURCE_COLLECTION })
  const denseField = desc.schema.fields.find((f) => f.name === 'dense_vector')
  const denseDim = Number(denseField?.type_params?.find((p) => p.key === 'dim')?.value ?? 1024)
  console.log(`dense_vector 维度: ${denseDim}`)

  // 删除旧集合（重建以支持 BM25）
  const exists = await railway.hasCollection({ collection_name: COLLECTION_NAME })
  if (exists.value) {
    await railway.dropCollection({ collection_name: COLLECTION_NAME })
    console.log(`已删除旧集合 "${COLLECTION_NAME}"`)
  }

  // 重建集合：videoTexts 开启分词器，sparse_vector 由 BM25 Function 自动生成
  await railway.createCollection({
    collection_name: COLLECTION_NAME,
    fields: [
      { name: 'id',                data_type: DataType.VarChar,       max_length: 64,    is_primary_key: true, autoID: false },
      { name: 'userId',            data_type: DataType.VarChar,       max_length: 128   },
      { name: 'account',           data_type: DataType.VarChar,       max_length: 256   },
      { name: 'platform',          data_type: DataType.VarChar,       max_length: 64    },
      { name: 'nickname',          data_type: DataType.VarChar,       max_length: 512   },
      { name: 'followerCount',     data_type: DataType.Int64          },
      { name: 'averagePlayCount',  data_type: DataType.Int64          },
      { name: 'lastPublishedTime', data_type: DataType.Int64          },
      { name: 'region',            data_type: DataType.VarChar,       max_length: 64    },
      { name: 'signature',         data_type: DataType.VarChar,       max_length: 2048  },
      {
        name: 'videoTexts',
        data_type: DataType.VarChar,
        max_length: 65535,
        enable_analyzer: true,         // 开启分词，BM25 的基础
        analyzer_params: { type: 'english' },
      },
      { name: 'email',             data_type: DataType.VarChar,       max_length: 256   },
      { name: 'dense_vector',      data_type: DataType.FloatVector,   dim: denseDim     },
      { name: 'sparse_vector',     data_type: DataType.SparseFloatVector                },  // 由 BM25 Function 自动填充
      { name: 'meta',              data_type: DataType.JSON           },
      { name: 'reserved1',         data_type: DataType.VarChar,       max_length: 1024  },
      { name: 'reserved2',         data_type: DataType.VarChar,       max_length: 1024  },
      { name: 'reserved3',         data_type: DataType.VarChar,       max_length: 1024  },
      { name: 'createdAt',         data_type: DataType.Int64          },
      { name: 'updatedAt',         data_type: DataType.Int64          },
      { name: 'language',          data_type: DataType.VarChar,       max_length: 32    },
      { name: 'ai_summary',        data_type: DataType.VarChar,       max_length: 65535 },
    ],
    // BM25 Function：videoTexts → sparse_vector（插入时自动计算）
    functions: [
      {
        name: 'bm25_fn',
        type: FunctionType.BM25,
        input_field_names: ['videoTexts'],
        output_field_names: ['sparse_vector'],
        params: {},
      },
    ],
  })
  console.log(`集合 "${COLLECTION_NAME}" 创建成功（含 BM25 Function）`)

  // dense_vector：HNSW + COSINE
  await railway.createIndex({
    collection_name: COLLECTION_NAME,
    field_name: 'dense_vector',
    index_name: 'dense_idx',
    index_type: 'HNSW',
    metric_type: 'COSINE',
    params: { M: 16, efConstruction: 200 },
  })

  // sparse_vector：BM25 自动管理，使用 SPARSE_INVERTED_INDEX + BM25
  await railway.createIndex({
    collection_name: COLLECTION_NAME,
    field_name: 'sparse_vector',
    index_name: 'sparse_idx',
    index_type: 'SPARSE_INVERTED_INDEX',
    metric_type: 'BM25',
  })

  await railway.loadCollection({ collection_name: COLLECTION_NAME })

  // 确认结果
  const final = await railway.describeCollection({ collection_name: COLLECTION_NAME })
  console.log(`\n字段总数: ${final.schema.fields.length}`)
  final.schema.fields.forEach((f) => {
    const dim = f.type_params?.find((p) => p.key === 'dim')?.value
    const analyzer = f.type_params?.find((p) => p.key === 'enable_analyzer')?.value
    console.log(` - ${f.name} (${f.data_type})${dim ? ` dim=${dim}` : ''}${analyzer ? ' [analyzer]' : ''}`)
  })
  console.log('\nFunctions:')
  final.schema.functions?.forEach((fn) => {
    console.log(` - ${fn.name}: ${fn.input_field_names?.join(',') ?? ''} → ${fn.output_field_names?.join(',') ?? ''}`)
  })
}

main().catch((err) => {
  console.error('失败:', err.message ?? err)
  process.exit(1)
})
