import 'dotenv/config';
import { createRailwayClient, createZillizClient } from './milvus.js';
const railway = createRailwayClient();
const zilliz = createZillizClient();
// Railway 实际行数（flush 后才可见）
await railway.flushSync({ collection_names: ['easykol_influencer_v1'] });
const s = await railway.getCollectionStatistics({ collection_name: 'easykol_influencer_v1' });
console.log('Railway 条数（flush后）:', s.stats.find((x) => x.key === 'row_count')?.value ?? '0');
// 检查 Zilliz 中有多少 YOUTUBE 记录的 dense_vector 为空
const res = await zilliz.query({
    collection_name: 'easykol_milvus_beta',
    filter: "platform == 'YOUTUBE'",
    output_fields: ['id', 'dense_vector'],
    limit: 10,
});
const nullVec = res.data.filter((r) => {
    const v = r.dense_vector;
    return !v || v.length === 0;
});
console.log(`\n抽查前 10 条 YOUTUBE 记录中，dense_vector 为空: ${nullVec.length} 条`);
res.data.forEach((r) => {
    const v = r.dense_vector;
    console.log(` id=${r.id} vector长度=${v?.length ?? 'null'}`);
});
