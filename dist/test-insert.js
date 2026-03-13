import 'dotenv/config';
import { createRailwayClient, createZillizClient } from './milvus.js';
const railway = createRailwayClient();
const zilliz = createZillizClient();
console.log('拉取 1 条测试数据...');
const res = await zilliz.query({
    collection_name: 'easykol_milvus_beta',
    filter: "platform == 'YOUTUBE'",
    output_fields: ['id', 'userId', 'account', 'platform', 'nickname', 'followerCount', 'averagePlayCount',
        'lastPublishedTime', 'region', 'signature', 'videoTexts', 'email', 'dense_vector', 'meta',
        'reserved1', 'reserved2', 'reserved3', 'createdAt', 'updatedAt', 'language'],
    limit: 1,
});
const row = res.data[0];
console.log('拉取成功，id:', row.id);
console.log('dense_vector length:', row.dense_vector?.length);
console.log('开始插入...');
const insertRes = await railway.insert({
    collection_name: 'easykol_influencer_v1',
    data: [{
            id: String(row.id ?? ''),
            userId: String(row.userId ?? ''),
            account: String(row.account ?? ''),
            platform: String(row.platform ?? ''),
            nickname: String(row.nickname ?? ''),
            followerCount: Number(row.followerCount ?? 0),
            averagePlayCount: Number(row.averagePlayCount ?? 0),
            lastPublishedTime: Number(row.lastPublishedTime ?? 0),
            region: String(row.region ?? ''),
            signature: String(row.signature ?? ''),
            videoTexts: String(row.videoTexts ?? ''),
            email: String(row.email ?? ''),
            dense_vector: row.dense_vector,
            meta: row.meta ?? {},
            reserved1: String(row.reserved1 ?? ''),
            reserved2: String(row.reserved2 ?? ''),
            reserved3: String(row.reserved3 ?? ''),
            createdAt: Number(row.createdAt ?? 0),
            updatedAt: Number(row.updatedAt ?? 0),
            language: String(row.language ?? ''),
            ai_summary: 'test summary',
        }],
});
console.log('插入结果:', JSON.stringify(insertRes));
