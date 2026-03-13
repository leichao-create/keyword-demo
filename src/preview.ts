import 'dotenv/config'
import { createRailwayClient } from './milvus.js'

const railway = createRailwayClient()
const COLLECTION = 'easykol_influencer_v1'

const res = await railway.query({
  collection_name: COLLECTION,
  output_fields: ['id', 'nickname', 'platform', 'account', 'region', 'followerCount', 'averagePlayCount', 'language', 'ai_summary'],
  limit: 100,
  offset: 0,
})

console.log(`取到 ${res.data.length} 条：\n`)
res.data.forEach((row, i) => {
  const summary = row.ai_summary ? String(row.ai_summary).slice(0, 60) + '...' : '(空)'
  console.log(
    `[${String(i + 1).padStart(4)}] ${String(row.nickname ?? '').padEnd(24)} | ${String(row.platform ?? '').padEnd(9)} | ${String(row.region ?? '').padEnd(4)} | 粉丝:${String(row.followerCount ?? 0).padStart(8)} | ${summary}`,
  )
})
