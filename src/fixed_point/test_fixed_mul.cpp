#include "FixedPoint.h"
#include <iostream>
#include <cassert>

void test_multiplication() {
    FixedPoint a(2, 8);  // 2.0 in fixed-point (Q8.8 format)
    FixedPoint b(3, 8);  // 3.0 in fixed-point (Q8.8 format)
    
    FixedPoint result = a * b;
    assert(result.toFloat() == 6.0);  // 2.0 * 3.0 should be 6.0
    
    std::cout << "Multiplication test passed!" << std::endl;
}

void test_multiplication_overflow() {
    FixedPoint a(1000, 8);   // 1000.0 in fixed-point (Q8.8 format)
    FixedPoint b(1000, 8);   // 1000.0 in fixed-point (Q8.8 format)
    
    FixedPoint result = a * b;
    assert(result.toFloat() == 255.0);  // Should saturate to max value due to overflow
    
    std::cout << "Multiplication overflow test passed!" << std::endl;
}

int main() {
    test_multiplication();
    test_multiplication_overflow();
    return 0;
}
