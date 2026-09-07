#include <spatialindex/SpatialIndex.h>
#include <spatialindex/RTree.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace SpatialIndex;

class MetricsVisitor final : public IVisitor {
public:
    explicit MetricsVisitor(const IShape& query) : m_query(query) {}

    uint64_t index_nodes = 0;
    uint64_t leaf_nodes = 0;
    uint64_t result_count = 0;
    uint64_t id_sum = 0;
    uint64_t id_xor = 0;
    double max_result_distance = 0.0;

    void visitNode(const INode& n) override {
        if (n.isLeaf()) ++leaf_nodes;
        else ++index_nodes;
    }

    void visitData(const IData& d) override {
        ++result_count;
        const uint64_t id = static_cast<uint64_t>(d.getIdentifier());
        id_sum += id;
        id_xor ^= id;
        IShape* shape = nullptr;
        d.getShape(&shape);
        if (shape != nullptr) {
            const double dist = m_query.getMinimumDistance(*shape);
            if (dist > max_result_distance) max_result_distance = dist;
            delete shape;
        }
    }

    void visitData(std::vector<const IData*>& v) override {
        for (const IData* d : v) if (d != nullptr) visitData(*d);
    }

private:
    const IShape& m_query;
};

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: platon_knn_metrics TREE_BASE INDEX_ID QUERY_TXT OUTPUT_CSV\n";
        return 2;
    }
    try {
        std::string tree_base = argv[1];
        const id_type index_id = static_cast<id_type>(std::stoll(argv[2]));
        const std::string query_path = argv[3];
        const std::string output_path = argv[4];

        std::unique_ptr<IStorageManager> storage(StorageManager::loadDiskStorageManager(tree_base));
        if (!storage) throw std::runtime_error("failed to load disk storage manager");
        std::unique_ptr<ISpatialIndex> tree(RTree::loadRTree(*storage, index_id));
        if (!tree) throw std::runtime_error("failed to load R-tree");
        if (!tree->isIndexValid()) throw std::runtime_error("loaded R-tree is invalid");

        std::ifstream in(query_path);
        if (!in) throw std::runtime_error("cannot open query file: " + query_path);
        std::ofstream out(output_path);
        if (!out) throw std::runtime_error("cannot open output file: " + output_path);
        out << "query_id,k,result_count,id_sum,id_xor,index_node_accesses,leaf_node_accesses,total_node_accesses,elapsed_us,returned_distance,max_result_distance\n";
        out << std::setprecision(17);

        long long query_id = 0;
        uint32_t k = 0;
        double x = 0.0, y = 0.0;
        uint64_t processed = 0;
        while (in >> query_id >> x >> y >> k) {
            double coords[2] = {x, y};
            Point q(coords, 2);
            MetricsVisitor visitor(q);
            const auto t0 = std::chrono::steady_clock::now();
            // libspatialindex::ISpatialIndex::nearestNeighborQuery returns void.
            // Derive the kth/returned distance from the visitor's returned data instead.
            tree->nearestNeighborQuery(k, q, visitor);
            const auto t1 = std::chrono::steady_clock::now();
            if (visitor.result_count < static_cast<uint64_t>(k)) {
                throw std::runtime_error("nearestNeighborQuery returned fewer than k results");
            }
            const double returned_distance = visitor.result_count > 0
                ? visitor.max_result_distance
                : std::numeric_limits<double>::quiet_NaN();
            const double elapsed_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
            const uint64_t total_nodes = visitor.index_nodes + visitor.leaf_nodes;
            out << query_id << ',' << k << ',' << visitor.result_count << ',' << visitor.id_sum << ',' << visitor.id_xor << ','
                << visitor.index_nodes << ',' << visitor.leaf_nodes << ',' << total_nodes << ',' << elapsed_us << ','
                << returned_distance << ',' << visitor.max_result_distance << '\n';
            ++processed;
        }
        std::cout << "PLATON_KNN_RESULT,queries=" << processed << ",valid_tree=1\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "PLATON_KNN_ERROR: " << e.what() << '\n';
        return 1;
    }
}
