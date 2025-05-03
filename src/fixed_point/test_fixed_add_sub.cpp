#include "FixedPoint.h"
#include <iostream>
#include <cassert>

void test_addition() {
    FixedPoint a(5, 8);  // 5.0 in fixed-point (Q8.8 format)
    FixedPoint b(3, 8);  // 3.0 in fixed-point (Q8.8 format)
    
    FixedPoint result = a + b;
    assert(result.toFloat() == 8.0);  // 5.0 + 3.0 should be 8.0
    
    std::cout << "Addition test passed!" << std::endl;
}

void test_subtraction() {
    FixedPoint a(10, 8);  // 10.0 in fixed-point (Q8.8 format)
    FixedPoint b(4, 8);   // 4.0 in fixed-point (Q8.8 format)
    
    FixedPoint result = a - b;
    assert(result.toFloat() == 6.0);  // 10.0 - 4.0 should be 6.0
    
    std::cout << "Subtraction test passed!" << std::endl;
}


int main() {
    FixedPoint a(3, 8);
    FixedPoint b(2, 8);
    FixedPoint c = a + b;
    FixedPoint d = a - b;

    std::cout << "a + b = ";
    c.print();  // Should print 5.0

    std::cout << "a - b = ";
    d.print();  // Should print 1.0

    return 0;
}