#include "colmap/scene/database_cache.h"
#include "colmap/util/logging.h"
#include <algorithm>
#include <iostream>
#include <sys/resource.h>
int main(int argc, char** argv) {
  colmap::InitializeGlog(argv);
  if (argc != 2) return 2;
  colmap::Database database(argv[1]);
  auto cache = colmap::DatabaseCache::Create(database, 15, false, {});
  auto graph = cache->CorrespondenceGraph();
  std::vector<colmap::image_t> ids;
  for (const auto& item : cache->Images()) ids.push_back(item.first);
  std::sort(ids.begin(), ids.end());
  uint64_t hash = 1469598103934665603ULL, total = 0;
  auto add = [&hash](uint64_t value) { hash ^= value; hash *= 1099511628211ULL; };
  for (auto id : ids) {
    const auto& image = cache->Image(id);
    add(id); add(image.NumPoints2D());
    for (colmap::point2D_t index = 0; index < image.NumPoints2D(); ++index) {
      auto range = graph->FindCorrespondences(id, index);
      for (auto c = range.beg; c != range.end; ++c) {
        add(index); add(c->image_id); add(c->point2D_idx); ++total;
      }
    }
  }
  rusage usage; getrusage(RUSAGE_SELF, &usage);
  std::cout << "GRAPH_PROOF {\"images\":" << cache->NumImages()
            << ",\"pairs\":" << graph->NumImagePairs()
            << ",\"directed_correspondences\":" << total
            << ",\"graph_hash\":" << hash
            << ",\"peak_rss_kib\":" << usage.ru_maxrss << "}" << std::endl;
}
