/*
 * @file cuNormalizer.h
 * @brief normalize the correlation surface
 *
 * cuNormalizeProcessor is an abstract class for processors to normalize the correlation surface.
 * It has different implementations wrt different image sizes.
 * cuNormalizeFixed<64/128/.../1024> use a shared memory accelerated algorithm, which are limited by the number of cuda threads in a block.
 * cuNormalizeSAT uses the sum area table based algorithm, which applies to any size (used for >1024).
 * cuNormalizer is a wrapper class which determines which processor to use.
 */

#ifndef __CUNORMALIZER_H
#define __CUNORMALIZER_H

#include "cuArrays.h"

namespace isce3::matchtemplate::pycuampcor {

/**
 * Abstract class interface for correlation surface normalization processor
 * with different implementations
 */
class cuNormalizeProcessor {
public:
    // default constructor and destructor
    cuNormalizeProcessor() {}
    ~cuNormalizeProcessor() {}
    // execute interface
    virtual void execute(cuArrays<float> * correlation, cuArrays<float> *reference, cuArrays<float> *secondary) = 0;
};

class cuNormalizer {
private:
    cuNormalizeProcessor *processor;
public:
     // disable the default constructor
     cuNormalizer() = delete;
     // constructor with the secondary dimension
     cuNormalizer(int secondaryNX, int secondaryNY, int count);
     // destructor
     ~cuNormalizer();
     // execute correlation surface normalization
     void execute(cuArrays<float> *correlation, cuArrays<float> *reference, cuArrays<float> *secondary);
};

class cuNormalizeSAT : public cuNormalizeProcessor
{
private:
    cuArrays<float> *referenceSum2;
    cuArrays<float> *secondarySAT;
    cuArrays<float> *secondarySAT2;

public:
    cuNormalizeSAT(int secondaryNX, int secondaryNY, int count);
    ~cuNormalizeSAT();
    void execute(cuArrays<float> * correlation, cuArrays<float> *reference, cuArrays<float> *search) override;
};

} // namespace

#endif
// end of file
