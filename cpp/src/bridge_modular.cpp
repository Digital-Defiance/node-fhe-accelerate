#include "modular_arithmetic.h"
#include <memory>

namespace fhe_accelerate {

// Factory function for creating ModularArithmetic instances
std::unique_ptr<ModularArithmetic> create_modular_arithmetic(uint64_t modulus) {
    return std::make_unique<ModularArithmetic>(modulus);
}

} // namespace fhe_accelerate
