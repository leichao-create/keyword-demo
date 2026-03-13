import 'dotenv/config';
import OpenAI from 'openai';
import { createRailwayClient } from './milvus.js';
const COLLECTION = process.env.MILVUS_COLLECTION ?? 'easykol_influencer_v1';
const TOP_K = 20;
const OUTPUT_FIELDS = [
    'id', 'userId', 'account', 'platform', 'nickname',
    'followerCount', 'averagePlayCount', 'region', 'language',
    'ai_summary', 'signature', 'meta', 'lastPublishedTime',
];
const openai = new OpenAI({
    baseURL: 'https://aihubmix.com/v1',
    apiKey: process.env.AIHUBMIX_API_KEY ?? '',
});
async function getEmbedding(text) {
    if (!process.env.AIHUBMIX_API_KEY)
        return null;
    try {
        const res = await openai.embeddings.create({
            model: process.env.EMBEDDING_MODEL ?? 'text-embedding-3-large',
            input: text,
        });
        return res.data[0].embedding;
    }
    catch (err) {
        console.error('[search] embedding error:', err);
        return null;
    }
}
function escapeLike(s) {
    return s.replace(/\\/g, '\\\\').replace(/%/g, '\\%').replace(/_/g, '\\_');
}
function buildSummaryFilter(keywords) {
    const terms = keywords.trim().split(/\s+/).filter(Boolean).slice(0, 5);
    if (!terms.length)
        return '';
    return terms.map(t => `ai_summary like '%${escapeLike(t)}%'`).join(' or ');
}
function normalizeRecord(r) {
    return {
        id: String(r.id ?? ''),
        userId: String(r.userId ?? ''),
        account: String(r.account ?? ''),
        platform: String(r.platform ?? ''),
        nickname: String(r.nickname ?? ''),
        followerCount: Number(r.followerCount ?? 0),
        averagePlayCount: Number(r.averagePlayCount ?? 0),
        region: String(r.region ?? ''),
        language: String(r.language ?? ''),
        ai_summary: String(r.ai_summary ?? ''),
        signature: String(r.signature ?? ''),
        meta: r.meta,
        sources: [],
        score: typeof r.score === 'number' ? r.score : undefined,
    };
}
function mergeResults(resultSets, limit) {
    const seen = new Map();
    for (const { hits, source } of resultSets) {
        hits.forEach((hit) => {
            const id = hit.id;
            if (!id)
                return;
            if (!seen.has(id)) {
                seen.set(id, { ...hit, sources: [source] });
            }
            else {
                seen.get(id).sources.push(source);
            }
        });
    }
    // 出现在越多路径 → 排前面；相同路径数按原顺序（BM25 score优先）
    return Array.from(seen.values())
        .sort((a, b) => b.sources.length - a.sources.length)
        .slice(0, limit);
}
export async function multiPathSearch(keywords, limit = 20) {
    const milvus = createRailwayClient();
    const paths = {};
    const embeddingResult = await Promise.resolve(getEmbedding(keywords)).then(v => ({ status: 'fulfilled', value: v })).catch(e => ({ status: 'rejected', reason: e }));
    const resultSets = [];
    const pendingJobs = [];
    // Path 1: 稠密向量语义搜索
    if (embeddingResult.status === 'fulfilled' && embeddingResult.value) {
        const vec = embeddingResult.value;
        pendingJobs.push(milvus
            .search({
            collection_name: COLLECTION,
            data: [vec],
            anns_field: 'dense_vector',
            limit: TOP_K,
            output_fields: OUTPUT_FIELDS,
            params: { ef: 200 },
        })
            .then((res) => {
            const hits = (res?.results ?? []).map(normalizeRecord);
            return { hits, source: 'dense' };
        })
            .catch((err) => {
            console.error('[search] dense error:', err);
            return null;
        }));
    }
    else {
        paths['dense'] = 0;
    }
    // Path 2: ai_summary 关键词匹配
    const summaryFilter = buildSummaryFilter(keywords);
    if (summaryFilter) {
        pendingJobs.push(milvus
            .query({
            collection_name: COLLECTION,
            filter: summaryFilter,
            output_fields: OUTPUT_FIELDS,
            limit: TOP_K,
        })
            .then((res) => {
            const rawData = Array.isArray(res) ? res : (res.data ?? []);
            const hits = rawData.map(normalizeRecord);
            return { hits, source: 'summary' };
        })
            .catch((err) => {
            console.error('[search] summary error:', err);
            return null;
        }));
    }
    else {
        paths['summary'] = 0;
    }
    const extras = await Promise.all(pendingJobs);
    for (const r of extras) {
        if (r) {
            resultSets.push(r);
            paths[r.source] = r.hits.length;
        }
    }
    const results = mergeResults(resultSets, limit);
    return { results, paths };
}
