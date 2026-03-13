import 'dotenv/config';
import { MilvusClient } from '@zilliz/milvus2-sdk-node';
const COLLECTION_NAME = 'easykol_influencer_v1';
async function main() {
    const railway = new MilvusClient({
        address: process.env.MILVUS_RAILWAY_ADDRESS ?? 'grpc-reverse-proxy-production-7da1.up.railway.app:443',
        ssl: true,
        timeout: 30000,
    });
    // 1. 列出所有集合（轻量操作）
    console.log('1. 获取集合列表...');
    const list = await railway.listCollections();
    console.log('集合列表:', list.data?.map((c) => c.name ?? c) ?? list);
    // 2. 检查 load 状态（不触发加载）
    console.log(`\n2. 检查 "${COLLECTION_NAME}" 加载状态...`);
    const loadState = await railway.getLoadState({ collection_name: COLLECTION_NAME });
    console.log('加载状态:', loadState);
    // 3. 获取集合统计（不需要 load）
    console.log('\n3. 获取集合统计...');
    const stats = await railway.getCollectionStatistics({ collection_name: COLLECTION_NAME });
    console.log('统计:', stats);
}
main().catch((err) => {
    console.error('失败:', err.message ?? err);
    process.exit(1);
});
