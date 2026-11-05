#include <spatialindex/SpatialIndex.h>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

using namespace SpatialIndex;

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "usage: native_dynamic_rtree_build records.txt tree_base capacity fill_factor page_size variant dimension\n";
        return 2;
    }
    try {
        const std::string records_path = argv[1];
        std::string tree_base = argv[2];
        const uint32_t capacity = static_cast<uint32_t>(std::stoul(argv[3]));
        const double fill_factor = std::stod(argv[4]);
        const uint32_t page_size = static_cast<uint32_t>(std::stoul(argv[5]));
        const std::string variant = argv[6];
        const uint32_t dimension = static_cast<uint32_t>(std::stoul(argv[7]));
        if (dimension != 2) throw std::runtime_error("only 2D supported");
        if (!(fill_factor > 0.0 && fill_factor < 1.0)) throw std::runtime_error("fill factor must be in (0,1)");
        RTree::RTreeVariant rv = RTree::RV_QUADRATIC;
        if (variant == "rstar") rv = RTree::RV_RSTAR;
        else if (variant == "quadratic") rv = RTree::RV_QUADRATIC;
        else if (variant == "linear") rv = RTree::RV_LINEAR;
        else throw std::runtime_error("variant must be rstar, quadratic, or linear");

        std::filesystem::remove(tree_base + ".idx");
        std::filesystem::remove(tree_base + ".dat");
        std::unique_ptr<IStorageManager> storage(StorageManager::createNewDiskStorageManager(tree_base, page_size));
        id_type index_id = -1;
        std::unique_ptr<ISpatialIndex> tree(RTree::createNewRTree(*storage, fill_factor, capacity, capacity, dimension, rv, index_id));
        if (!tree) throw std::runtime_error("failed to create tree");
        std::ifstream in(records_path);
        if (!in) throw std::runtime_error("cannot open records file");
        auto t0 = std::chrono::steady_clock::now();
        int op = 0; long long raw_id = 0; double xmin=0,ymin=0,xmax=0,ymax=0; uint64_t count=0;
        while (in >> op >> raw_id >> xmin >> ymin >> xmax >> ymax) {
            if (op != 1) throw std::runtime_error("unexpected record opcode");
            double low[2] = {xmin,ymin}; double high[2] = {xmax,ymax}; Region region(low,high,2);
            tree->insertData(0,nullptr,region,static_cast<id_type>(raw_id)); ++count;
        }
        tree->flush();
        bool valid = tree->isIndexValid();
        double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
        std::cout << std::setprecision(17)
                  << "DYNAMIC_RTREE_BUILD_RESULT,index_identifier=" << index_id
                  << ",variant=" << variant << ",capacity=" << capacity
                  << ",objects=" << count << ",valid_tree=" << (valid?1:0)
                  << ",build_seconds=" << seconds << "\n";
        return valid ? 0 : 3;
    } catch (const std::exception& e) {
        std::cerr << "native_dynamic_rtree_build error: " << e.what() << "\n";
        return 1;
    }
}
