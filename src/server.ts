import 'dotenv/config'
import path from 'path'
import { fileURLToPath } from 'url'
import Fastify from 'fastify'
import cors from '@fastify/cors'
import staticPlugin from '@fastify/static'
import { multiPathSearch } from './search.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

const app = Fastify({ logger: { level: 'info' } })

await app.register(cors, { origin: true })
await app.register(staticPlugin, {
  root: path.join(__dirname, '../public'),
  prefix: '/',
})

app.post<{ Body: { keywords: string; limit?: number } }>(
  '/api/search',
  {
    schema: {
      body: {
        type: 'object',
        required: ['keywords'],
        properties: {
          keywords: { type: 'string', minLength: 1 },
          limit: { type: 'number', default: 20, minimum: 1, maximum: 100 },
        },
      },
    },
  },
  async (request, reply) => {
    const { keywords, limit = 20 } = request.body
    try {
      const data = await multiPathSearch(keywords.trim(), limit)
      return data
    } catch (err) {
      app.log.error(err)
      reply.status(500)
      return { error: '搜索失败，请稍后重试', results: [], paths: {} }
    }
  },
)

const port = Number(process.env.PORT ?? 3000)
await app.listen({ port, host: '0.0.0.0' })
console.log(`\n🚀 服务已启动: http://localhost:${port}\n`)
