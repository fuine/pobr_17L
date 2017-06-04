#include "Classifier.hpp"

// I know that this is REALLY REALLY NOT the way it should be done.....


/*
 * Decide if given segment is a letter 'C'
 */
bool classify_c(const Features& f) {
    if (f.central_moments[2][1] <= 173907.0) {
        if (f.central_moments[3][0] <= -36622552.0) {
            return f.central_moments[0][3] > 28956850.0;
        }
        else {
            return false;
        }
    }
    else {
        if (f.normal_moments[0][3] <= 28597900.0) {
            if (f.central_moments[0][3] <= 19892.4395) {
                return false;
            }
            else {
                return f.central_moments[0][3] <= 589551.5;
            }
        }
        else {
            if (f.central_moments[1][2] <= -1852275.0) {
                return f.coeffs[2] <= 1.0381;
            }
            else {
                return false;
            }
        }
    }
}

/*
 * Decide if given segment is a logo
 */
bool classify_logo(const Features& f) {
    if (f.central_moments[2][1] <= -513866.0) {
        if (f.coeffs[2] <= 1.0842) {
            if (f.central_moments[1][2] <= -1656670.0) {
                return f.central_moments[1][1] <= -529385.5;
            }
            else {
                return true;
            }
        }
        else {
            if (f.coeffs[3] <= 0.4429) {
                return false;
            }
            else {
                return f.normal_moments[0][1] > 210619.5;
            }
        }
    }
    else {
        if (f.central_moments[2][1] <= -194529.5) {
            return (f.central_moments[2][1] > -229043.5);
        }
        else {
            if (f.coeffs[2] <= 0.3971) {
                return f.central_moments[0][3] > 209793.5;
            }
            else {
                return false;
            }
        }
    }
}

bool classify(const Features& f) {
    return classify_c(f) || classify_logo(f);
}
