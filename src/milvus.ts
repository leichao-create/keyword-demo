import { MilvusClient, DataType } from '@zilliz/milvus2-sdk-node'

export function createRailwayClient(): MilvusClient {
  return new MilvusClient({
    address: process.env.MILVUS_RAILWAY_ADDRESS ?? 'grpc-reverse-proxy-production-7da1.up.railway.app:443',
    ssl: true,
    timeout: 120000,
  })
}

export function createZillizClient(): MilvusClient {
  return new MilvusClient({
    address: process.env.MILVUS_ZILLIZ_ADDRESS ?? 'in03-4751ecda463a927.serverless.gcp-us-west1.cloud.zilliz.com:443',
    ssl: true,
    username: process.env.MILVUS_ZILLIZ_USERNAME ?? '',
    password: process.env.MILVUS_ZILLIZ_PASSWORD ?? '',
    timeout: 180000,
  })
}

export async function describeCollection(client: MilvusClient, collectionName: string) {
  await client.loadCollection({ collection_name: collectionName })
  const desc = await client.describeCollection({ collection_name: collectionName })
  const stats = await client.getCollectionStatistics({ collection_name: collectionName })
  const rowCount = stats.stats.find((s) => s.key === 'row_count')?.value ?? '未知'
  return { fields: desc.schema.fields, rowCount }
}

export { DataType }
