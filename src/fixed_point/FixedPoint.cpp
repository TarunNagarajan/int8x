#include "FixedPoint.h"
#include <iostream>
#include <cmath> // Included, though not strictly necessary for the current implementation.

FixedPoint::FixedPoint() : value(0), fraction_bits(8) {}

FixedPoint::FixedPoint(int val, int frac_bits) : fraction_bits(frac_bits) {
    value = val * (1 << fraction_bits); // Scale the integer value to the fixed-point representation.
}

FixedPoint FixedPoint::operator+(const FixedPoint& other) const {
    if (fraction_bits != other.fraction_bits) {
        throw std::invalid_argument("Fraction bits must be the same.");
    }
    int result = value + other.value;
    return FixedPoint(result, fraction_bits);
}

FixedPoint FixedPoint::operator-(const FixedPoint& other) const {
    if (fraction_bits != other.fraction_bits) {
        throw std::invalid_argument("Fraction bits must be the same.");
    }
    int result = value - other.value;
    return FixedPoint(result, fraction_bits);
}

FixedPoint FixedPoint::operator*(const FixedPoint& other) const {
    if (fraction_bits != other.fraction_bits) {
        throw std::invalid_argument("Fraction bits must be the same.");
    }
    int result = (static_cast<long long>(value) * other.value) >> fraction_bits; // Use long long for intermediate result to avoid overflow.
    result = saturate(result); // Apply saturation to the result.
    return FixedPoint(result, fraction_bits);
}

FixedPoint FixedPoint::operator/(const FixedPoint& other) const {
    if (fraction_bits != other.fraction_bits) {
        throw std::invalid_argument("Fraction bits must be the same.");
    }
    if (other.value == 0) {
        throw std::overflow_error("Division by zero.");
    }
    int result = (static_cast<long long>(value) << fraction_bits) / other.value; // Use long long to avoid intermediate overflow.
    result = saturate(result); // Apply saturation.
    return FixedPoint(result, fraction_bits);
}

int FixedPoint::saturate(int result) const {
    int max_val = (1 << (31 - fraction_bits)) - 1;
    int min_val = -(1 << (31 - fraction_bits));

    if (result > max_val) {
        return max_val;
    }
    if (result < min_val) {
        return min_val;
    }
    return result;
}

int FixedPoint::toInt() const {
    return value >> fraction_bits; // Right shift to get the integer part.
}

float FixedPoint::toFloat() const {
    return static_cast<float>(value) / (1 << fraction_bits); // Convert to float by dividing by 2^fraction_bits.
}

void FixedPoint::print() const {
    std::cout << toFloat() << std::endl;
}