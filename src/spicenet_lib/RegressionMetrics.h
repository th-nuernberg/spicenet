//
// Created by Fabian on 06.02.2025.
//

#ifndef SPICENET_CPP_REGRESSIONMETRICS_H
#define SPICENET_CPP_REGRESSIONMETRICS_H

#include <type_traits>
#include <vector>

inline double bias(double predicted, double actual) {
    return predicted - actual;
}

inline double mean(const std::vector<double>& vector) {
    double sum = 0;
    for (auto test: vector) {
        sum += test;
    }
    return sum / vector.size();
}

inline double median(std::vector<double> vector) {
    std::sort(vector.begin(), vector.end());

    if (vector.size() % 2 != 0)
        return (double) vector[vector.size() / 2];

    return (double) (vector[(vector.size() - 1) / 2] + vector[vector.size() / 2]) / 2.0;
}

template<typename Iterator>
inline double meanBias(Iterator beginPredicted,
                       Iterator endPredicted,
                       Iterator beginActual,
                       Iterator endActual) {
    static_assert(std::is_arithmetic<typename std::iterator_traits<Iterator>::value_type>::value,
                  "Iterator must point to a numeric type!");

    double result = 0;
    double count = 0;
    while ((beginActual != endActual) && (beginPredicted != endPredicted)) {
        result += bias(*beginPredicted, *beginActual);
        ++count;
        ++beginPredicted;
        ++beginActual;
    }
    return result / count;
}

template<typename Iterator>
inline double medianBias(Iterator beginPredicted,
                         Iterator endPredicted,
                         Iterator beginActual,
                         Iterator endActual) {
    static_assert(std::is_arithmetic<typename std::iterator_traits<Iterator>::value_type>::value,
                  "Iterator must point to a numeric type!");
    std::vector<double> buffer;
    while ((beginActual != endActual) && (beginPredicted != endPredicted)) {
        buffer.push_back(bias(*beginPredicted, *beginActual));
        ++beginPredicted;
        ++beginActual;
    }
    return median(buffer);
}

template<typename Iterator>
inline double standardDeviationBias(Iterator beginPredicted,
                                    Iterator endPredicted,
                                    Iterator beginActual,
                                    Iterator endActual) {
    static_assert(std::is_arithmetic<typename std::iterator_traits<Iterator>::value_type>::value,
                  "Iterator must point to a numeric type!");
    std::vector<double> buffer;
    while ((beginActual != endActual) && (beginPredicted != endPredicted)) {
        buffer.push_back(bias(*beginPredicted, *beginActual));
        ++beginPredicted;
        ++beginActual;
    }
    auto meanValue = mean(buffer);
    double sum = 0;
    for (auto x: buffer) {
        sum += pow(x - meanValue, 2);
    }
    sum *= 1.0 / ((double) buffer.size());
    return sqrt(sum);
}

template<typename Iterator>
inline double meanAbsoluteGrossError(Iterator beginPredicted,
                                     Iterator endPredicted,
                                     Iterator beginActual,
                                     Iterator endActual) {
    static_assert(std::is_arithmetic<typename std::iterator_traits<Iterator>::value_type>::value,
                  "Iterator must point to a numeric type!");
    std::vector<double> buffer;
    while ((beginActual != endActual) && (beginPredicted != endPredicted)) {
        buffer.push_back(abs(bias(*beginPredicted, *beginActual)));
        ++beginPredicted;
        ++beginActual;
    }
    return mean(buffer);
}

template<typename Iterator>
inline double meanSquaredError(Iterator beginPredicted,
                               Iterator endPredicted,
                               Iterator beginActual,
                               Iterator endActual) {
    static_assert(std::is_arithmetic<typename std::iterator_traits<Iterator>::value_type>::value,
                  "Iterator must point to a numeric type!");
    std::vector<double> buffer;
    while ((beginActual != endActual) && (beginPredicted != endPredicted)) {
        buffer.push_back(pow(bias(*beginPredicted, *beginActual), 2));
        ++beginPredicted;
        ++beginActual;
    }
    return mean(buffer);
}

template<typename Iterator>
inline double rootMeanSquaredError(Iterator beginPredicted,
                                   Iterator endPredicted,
                                   Iterator beginActual,
                                   Iterator endActual) {
    static_assert(std::is_arithmetic<typename std::iterator_traits<Iterator>::value_type>::value,
                  "Iterator must point to a numeric type!");
    return sqrt(meanSquaredError(beginPredicted, endPredicted, beginActual, endActual));
}

template<typename Iterator>
inline double centeredMeanSquareDifference(Iterator beginPredicted,
                                           Iterator endPredicted,
                                           Iterator beginActual,
                                           Iterator endActual) {
    static_assert(std::is_arithmetic<typename std::iterator_traits<Iterator>::value_type>::value,
                  "Iterator must point to a numeric type!");
    std::vector<double> bufferPredicted;
    std::vector<double> bufferActual;
    while ((beginActual != endActual) && (beginPredicted != endPredicted)) {
        bufferPredicted.push_back(*beginPredicted);
        bufferActual.push_back(*beginActual);
        ++beginPredicted;
        ++beginActual;
    }
    double meanPredicted = mean(bufferPredicted);
    double meanActual = mean(bufferActual);
    std::vector<double> buffer(bufferPredicted.size());
    for (unsigned int i = 0; i < bufferPredicted.size(); ++i) {
        buffer.at(i) = pow(
                (bufferPredicted.at(i) - meanPredicted) -
                (bufferActual.at(i) - meanActual), 2);
    }
    return mean(buffer);
}

template<typename Iterator>
inline double centeredRootMeanSquaredError(Iterator beginPredicted,
                                           Iterator endPredicted,
                                           Iterator beginActual,
                                           Iterator endActual) {
    static_assert(std::is_arithmetic<typename std::iterator_traits<Iterator>::value_type>::value,
                  "Iterator must point to a numeric type!");
    return sqrt(centeredMeanSquareDifference(beginPredicted, endPredicted, beginActual, endActual));
}

template<typename Iterator>
inline double meanNormalizedBias(Iterator beginPredicted,
                                 Iterator endPredicted,
                                 Iterator beginActual,
                                 Iterator endActual) {
    static_assert(std::is_arithmetic<typename std::iterator_traits<Iterator>::value_type>::value,
                  "Iterator must point to a numeric type!");
    std::vector<double> buffer;
    while ((beginActual != endActual) && (beginPredicted != endPredicted)) {
        buffer.push_back(*beginPredicted / *beginActual);
        ++beginPredicted;
        ++beginActual;
    }
    return mean(buffer) - 1;
}

template<typename Iterator>
inline double meanNormalizedGrossError(Iterator beginPredicted,
                                       Iterator endPredicted,
                                       Iterator beginActual,
                                       Iterator endActual) {
    static_assert(std::is_arithmetic<typename std::iterator_traits<Iterator>::value_type>::value,
                  "Iterator must point to a numeric type!");
    std::vector<double> buffer;
    while ((beginActual != endActual) && (beginPredicted != endPredicted)) {
        buffer.push_back(abs(*beginPredicted - *beginActual) / *beginActual);
        ++beginPredicted;
        ++beginActual;
    }
    return mean(buffer);
}

template<typename Iterator>
inline double normalizedMeanBias(Iterator beginPredicted,
                                 Iterator endPredicted,
                                 Iterator beginActual,
                                 Iterator endActual) {
    static_assert(std::is_arithmetic<typename std::iterator_traits<Iterator>::value_type>::value,
                  "Iterator must point to a numeric type!");
    std::vector<double> bufferPredicted;
    std::vector<double> bufferActual;
    while ((beginActual != endActual) && (beginPredicted != endPredicted)) {
        bufferPredicted.push_back(*beginPredicted);
        bufferActual.push_back(*beginActual);
        ++beginPredicted;
        ++beginActual;
    }
    return (mean(bufferPredicted) / mean(bufferActual)) - 1;
}

template<typename Iterator>
inline double normalizedMeanError(Iterator beginPredicted,
                                  Iterator endPredicted,
                                  Iterator beginActual,
                                  Iterator endActual) {
    static_assert(std::is_arithmetic<typename std::iterator_traits<Iterator>::value_type>::value,
                  "Iterator must point to a numeric type!");
    auto temp = meanAbsoluteGrossError(beginPredicted, endPredicted, beginActual, endActual);
    std::vector<double> bufferActual;
    while ((beginActual != endActual) && (beginPredicted != endPredicted)) {
        bufferActual.push_back(*beginActual);
        ++beginPredicted;
        ++beginActual;
    }

    return temp / mean(bufferActual);
}

template<typename Iterator>
inline double fractionalBias(Iterator beginPredicted,
                             Iterator endPredicted,
                             Iterator beginActual,
                             Iterator endActual) {
    static_assert(std::is_arithmetic<typename std::iterator_traits<Iterator>::value_type>::value,
                  "Iterator must point to a numeric type!");
    double sum = 0;
    double count = 0;
    while ((beginActual != endActual) && (beginPredicted != endPredicted)) {
        sum += (*beginPredicted - *beginActual) / (*beginPredicted + *beginActual);
        ++count;
        ++beginPredicted;
        ++beginActual;
    }
    return 2.0 / count * sum;
}

template<typename Iterator>
inline double fractionalGrossError(Iterator beginPredicted,
                                   Iterator endPredicted,
                                   Iterator beginActual,
                                   Iterator endActual) {
    static_assert(std::is_arithmetic<typename std::iterator_traits<Iterator>::value_type>::value,
                  "Iterator must point to a numeric type!");
    double sum = 0;
    double count = 0;
    while ((beginActual != endActual) && (beginPredicted != endPredicted)) {
        sum += abs(*beginPredicted - *beginActual) / (*beginPredicted + *beginActual);
        ++count;
        ++beginPredicted;
        ++beginActual;
    }
    return 2.0 / count * sum;
}

template<typename Iterator>
inline double theilsUi(Iterator beginPredicted,
                       Iterator endPredicted,
                       Iterator beginActual,
                       Iterator endActual) {
    static_assert(std::is_arithmetic<typename std::iterator_traits<Iterator>::value_type>::value,
                  "Iterator must point to a numeric type!");
    double rootMeanSquaredErrorValue = rootMeanSquaredError(beginPredicted, endPredicted, beginActual, endActual);
    std::vector<double> bufferPredicted;
    std::vector<double> bufferActual;
    while ((beginActual != endActual) && (beginPredicted != endPredicted)) {
        bufferPredicted.push_back(pow(*beginPredicted, 2));
        bufferActual.push_back(pow(*beginActual, 2));
        ++beginPredicted;
        ++beginActual;
    }

    return rootMeanSquaredErrorValue / (sqrt(mean(bufferPredicted)) + sqrt(mean(bufferActual)));
}

template<typename Iterator>
inline double indexOfAgreement(Iterator beginPredicted,
                               Iterator endPredicted,
                               Iterator beginActual,
                               Iterator endActual) {
    static_assert(std::is_arithmetic<typename std::iterator_traits<Iterator>::value_type>::value,
                  "Iterator must point to a numeric type!");
    double meanSquaredErrorValue = meanSquaredError(beginPredicted, endPredicted, beginActual, endActual);
    std::vector<double> bufferPredicted;
    std::vector<double> bufferActual;
    while ((beginActual != endActual) && (beginPredicted != endPredicted)) {
        bufferPredicted.push_back(*beginPredicted);
        bufferActual.push_back(*beginActual);
        ++beginPredicted;
        ++beginActual;
    }
    double meanActual = mean(bufferActual);

    double sum = 0;
    for (unsigned int i = 0; i < bufferPredicted.size(); ++i) {
        sum += pow(abs(bufferPredicted.at(i) - meanActual)
                   + abs(bufferActual.at(i) - meanActual), 2);
    }

    return 1 - ((meanSquaredErrorValue * bufferPredicted.size()) / sum);
}

template<typename Iterator>
inline std::map<std::string, double> metricsMap(Iterator beginPredicted,
                                                Iterator endPredicted,
                                                Iterator beginActual,
                                                Iterator endActual) {
    static_assert(std::is_arithmetic<typename std::iterator_traits<Iterator>::value_type>::value,
                  "Iterator must point to a numeric type!");

    return {
            {"Mean Bias",                            meanBias(beginPredicted, endPredicted, beginActual, endActual)},
            {"Median Bias",                          medianBias(beginPredicted, endPredicted, beginActual, endActual)},
            {"Standard Deviation of Bias",           standardDeviationBias(beginPredicted, endPredicted, beginActual,
                                                                           endActual)},
            {"Mean Absolute Gross Error",            meanAbsoluteGrossError(beginPredicted, endPredicted, beginActual,
                                                                            endActual)},
            {"Mean Squared Error",                   meanSquaredError(beginPredicted, endPredicted, beginActual,
                                                                      endActual)},
            {"Root Mean Squared Error",              rootMeanSquaredError(beginPredicted, endPredicted, beginActual,
                                                                          endActual)},
            {"Centered Mean Square Difference",      centeredMeanSquareDifference(beginPredicted, endPredicted,
                                                                                  beginActual, endActual)},
            {"Centered Root Mean Square Difference", centeredRootMeanSquaredError(beginPredicted, endPredicted,
                                                                                  beginActual, endActual)},
            {"Mean Normalized Bias",                 meanNormalizedBias(beginPredicted, endPredicted, beginActual,
                                                                        endActual)},
            {"Mean Normalized Gross Error",          meanNormalizedGrossError(beginPredicted, endPredicted, beginActual,
                                                                              endActual)},
            {"Normalized Mean Bias",                 normalizedMeanBias(beginPredicted, endPredicted, beginActual,
                                                                        endActual)},
            {"Normalized Mean Error",                normalizedMeanError(beginPredicted, endPredicted, beginActual,
                                                                         endActual)},
            {"Fractional Bias",                      fractionalBias(beginPredicted, endPredicted, beginActual,
                                                                    endActual)},
            {"Fractional Gross Error",               fractionalGrossError(beginPredicted, endPredicted, beginActual,
                                                                          endActual)},
            {"Theil’s UI",                           theilsUi(beginPredicted, endPredicted, beginActual, endActual)},
            {"Index of agreement",                   indexOfAgreement(beginPredicted, endPredicted, beginActual,
                                                                      endActual)},
    };
}

#endif //SPICENET_CPP_REGRESSIONMETRICS_H
