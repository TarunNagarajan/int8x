#ifndef FIXEDPOINT_H
#define FIXEDPOINT_H

#include <iostream> 
#include <stdexcept> 

class FixedPoint {
public:
    // Default constructor. Initializes the FixedPoint object with a default value (likely 0)
    // and a reasonable default for the number of fractional bits. Consider making the
    // default number of fractional bits a static constant for consistency.
    FixedPoint();

    // Constructor to initialize the FixedPoint object with a given integer value and
    // the number of bits to be used for the fractional part. The 'value' here will
    // likely be left-shifted by 'fraction_bits' to create the internal representation.
    FixedPoint(int value, int fraction_bits);

    // Overload of the addition operator. Ensures that operations between FixedPoint
    // objects maintain the same number of fractional bits. Consider throwing an exception
    // if the 'other' FixedPoint has a different 'fraction_bits'.
    FixedPoint operator+(const FixedPoint& other) const;

    // Overload of the subtraction operator. Similar considerations to the addition operator
    // regarding consistent 'fraction_bits'.
    FixedPoint operator-(const FixedPoint& other) const;

    // Overload of the multiplication operator. This will require careful handling of the
    // fractional bits in the result (the number of fractional bits will effectively double
    // and then needs to be shifted back). Consider potential overflow issues.
    FixedPoint operator*(const FixedPoint& other) const;

    // Overload of the division operator. This is the most complex operation and needs
    // careful implementation to handle fractional bits and potential division by zero.
    // Think about the precision implications and potential for truncation.
    FixedPoint operator/(const FixedPoint& other) const;

    // Converts the FixedPoint value to its nearest integer representation. This will likely
    // involve a right shift and potentially rounding.
    int toInt() const;

    // Converts the FixedPoint value to a floating-point number (float). Useful for
    // interoperability with standard floating-point operations or for display. Be aware
    // of potential precision loss during the conversion.
    float toFloat() const;

    // Prints the FixedPoint value to the standard output. Consider formatting the output
    // to clearly show both the integer and fractional parts (perhaps as a decimal).
    void print() const;

private:
    // The underlying integer value that represents the fixed-point number. The actual
    // numerical value is 'value / (2^fraction_bits)'.
    int value;

    // The number of bits dedicated to representing the fractional part of the number.
    // This determines the precision of the fixed-point representation.
    int fraction_bits;

    // Private helper method for handling potential overflow or underflow after arithmetic
    // operations. Saturation ensures the result stays within the representable range.
    // The specific implementation of saturation will depend on the intended range of the
    // FixedPoint numbers.
    int saturate(int result) const;
};

#endif // FIXEDPOINT_H