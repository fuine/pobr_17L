#ifndef CLASSIFIER_HPP
#define CLASSIFIER_HPP
#include "Moments.hpp"

/*
 * Classify segment based on its features -- returns true if segment is either a
 * letter 'C' or the blue part of the logo itself
 */
bool classify(const Features& f);

#endif /* ifndef CLASSIFIER_HPP */
