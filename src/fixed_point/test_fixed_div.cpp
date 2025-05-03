#include "FixedPoint.h"
#include <iostream>
#include <cassert>

void test_division() {
    FixedPoint a(6, 8);  // 6.0 in fixed-point (Q8.8 format)
    FixedPoint b(3, 8);  // 3.0 in fixed-point (Q8.8 format)
    
    FixedPoint result = a / b;
    assert(result.toFloat() == 2.0);  // 6.0 / 3.0 should be 2.0
    
    std::cout << "Division test passed!" << std::endl;
}

void test_division_by_zero() {
    FixedPoint a(6, 8);  // 6.0 in fixed-point (Q8.8 format)
    FixedPoint b(0, 8);  // 0.0 in fixed-point (Q8.8 format)
    
    try {
        FixedPoint result = a / b;
        assert(false);  // Should throw exception
    } catch (const std::overflow_error& e) {
        std::cout << "Division by zero caught as expected!" << std::endl;
    }
}

int main() {
    test_division();
    test_division_by_zero();
    return 0;
}
