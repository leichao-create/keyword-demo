import 'dotenv/config'
import OpenAI from 'openai'
import { createRailwayClient } from './milvus.js'

const TARGET_COLLECTION = 'easykol_influencer_v1'
const AI_CONCURRENCY = 5
const PAGE_SIZE = 500

const ai = new OpenAI({
  apiKey: process.env.AIHUBMIX_API_KEY ?? '',
  baseURL: 'https://aihubmix.com/v1',
})
const MODEL = process.env.AIHUBMIX_MODEL ?? 'gemini-2.5-flash'

const SUMMARY_PROMPT = (videoTexts: string) =>
  `Act as a professional media content analyst. Please analyze the provided list of video titles and summarize the blogger's niche and content.

### Constraints:
1. Language: Always respond in English, regardless of the input language.
2. Format: Provide a single, concise sentence ending with a period.
3. Content: Accurately summarize the blogger's "niche/field" and "core content themes."
4. Tone: Professional and direct.

### Input Data:
${videoTexts}`

async function generateSummary(videoTexts: string): Promise<string> {
  const MAX_RETRIES = 4
  let maxTokens = 512

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      const res = await ai.chat.completions.create({
        model: MODEL,
        messages: [{ role: 'user', content: SUMMARY_PROMPT(videoTexts) }],
        max_tokens: maxTokens,
      })

      const choice = res.choices[0]
      const content = choice?.message?.content?.trim() ?? ''

      // 被 token 截断，扩大 max_tokens 重试
      if (choice?.finish_reason === 'length') {
        maxTokens = Math.min(maxTokens * 2, 2048)
        console.log(`    ↑ finish_reason=length，扩大 max_tokens 到 ${maxTokens}，重试 ${attempt}/${MAX_RETRIES}`)
        continue
      }

      // 返回空内容，等待后重试
      if (!content) {
        const delay = attempt * 3000
        console.log(`    ↑ AI 返回空，${delay / 1000}s 后重试 ${attempt}/${MAX_RETRIES}`)
        await new Promise((r) => setTimeout(r, delay))
        continue
      }

      return content
    } catch (err: any) {
      const delay = attempt * 3000
      const msg = String(err?.message ?? '').slice(0, 80)
      if (attempt < MAX_RETRIES) {
        console.log(`    ↑ 请求错误: ${msg}，${delay / 1000}s 后重试 ${attempt}/${MAX_RETRIES}`)
        await new Promise((r) => setTimeout(r, delay))
      } else {
        throw err
      }
    }
  }
  return ''
}

function isTruncated(summary: string): boolean {
  if (!summary) return true
  const trimmed = summary.trim()
  // 句子没有以标点结尾，视为截断
  return !/[.!?。！？]$/.test(trimmed)
}

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

async function main() {
  const railway = createRailwayClient()
  await railway.loadCollection({ collection_name: TARGET_COLLECTION })

  // 分页读取所有记录
  console.log('读取 Railway 集合中所有记录...')
  const allRows: any[] = []
  let page = 0
  while (true) {
    const res = await railway.query({
      collection_name: TARGET_COLLECTION,
      output_fields: ['id', 'nickname', 'videoTexts', 'ai_summary'],
      limit: PAGE_SIZE,
      offset: page * PAGE_SIZE,
    })
    if (!res.data?.length) break
    allRows.push(...res.data)
    console.log(`  已读取 ${allRows.length} 条...`)
    if (res.data.length < PAGE_SIZE) break
    page++
  }
  console.log(`共读取 ${allRows.length} 条记录\n`)

  // 找出需要修复的记录
  const truncated = allRows.filter((r) => isTruncated(String(r.ai_summary ?? '')))
  console.log(`发现 ${truncated.length} 条摘要被截断，开始修复...\n`)

  if (truncated.length === 0) {
    console.log('没有需要修复的记录，退出。')
    return
  }

  // 并发重新生成
  const summaries = await mapWithConcurrency(
    truncated,
    async (row, i) => {
      const nick = String(row.nickname ?? '').slice(0, 20).padEnd(20)
      try {
        const summary = await generateSummary(String(row.videoTexts ?? ''))
        console.log(`  [${String(i + 1).padStart(4)}/${truncated.length}] ${nick}`)
        console.log(`           旧: ${String(row.ai_summary ?? '').slice(0, 80)}`)
        console.log(`           新: ${summary || '[WARN: AI 返回空]'}\n`)
        return summary
      } catch (err: any) {
        console.log(`  [${String(i + 1).padStart(4)}] ${nick} → [ERROR: ${String(err?.message ?? '').slice(0, 60)}]`)
        return String(row.ai_summary ?? '')
      }
    },
    AI_CONCURRENCY,
  )

  // 读取完整记录（upsert 需要所有字段，包括向量）
  console.log('读取完整字段以准备 upsert...')
  const ids = truncated.map((r) => String(r.id))
  const idFilter = `id in [${ids.map((id) => `"${id}"`).join(', ')}]`
  const fullRes = await railway.query({
    collection_name: TARGET_COLLECTION,
    filter: idFilter,
    output_fields: ['*'],
    limit: ids.length,
  })

  const fullMap = new Map(fullRes.data.map((r: any) => [String(r.id), r]))

  const records = truncated
    .map((row, i) => {
      const full = fullMap.get(String(row.id))
      if (!full) return null
      return { ...full, ai_summary: summaries[i] }
    })
    .filter(Boolean)

  // Upsert 回 Railway
  console.log(`\n开始 upsert ${records.length} 条修复数据...`)
  const BATCH = 50
  for (let i = 0; i < records.length; i += BATCH) {
    const batch = records.slice(i, i + BATCH)
    await railway.upsert({ collection_name: TARGET_COLLECTION, data: batch })
    console.log(`  ✓ upsert ${i + batch.length}/${records.length}`)
  }

  await railway.flushSync({ collection_names: [TARGET_COLLECTION] })
  console.log(`\n✅ 修复完成，共更新 ${records.length} 条摘要`)
}

main().catch((err) => {
  console.error('修复失败:', err.message ?? err)
  process.exit(1)
})
