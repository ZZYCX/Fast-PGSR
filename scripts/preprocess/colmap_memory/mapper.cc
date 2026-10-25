#include "colmap/exe/sfm.h"
#include "colmap/util/logging.h"
int main(int argc, char** argv) {
  colmap::InitializeGlog(argv);
  return colmap::RunMapper(argc, argv);
}
