#include "my_torch/Metrics.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>

#include "my_torch/Error.hpp"

namespace my_torch {

namespace {

constexpr Scalar logFloor = 1e-15;

void requireSameShape(const char *name, const Matrix &probabilities, const Matrix &targets)
{
    if (probabilities.rows() == 0 || probabilities.cols() == 0)
        throw Error(std::string(name) + ": empty input");
    if (probabilities.rows() != targets.rows() || probabilities.cols() != targets.cols())
        throw Error(std::string(name) + ": " + std::to_string(probabilities.rows()) + "x" +
                    std::to_string(probabilities.cols()) + " probabilities for " +
                    std::to_string(targets.rows()) + "x" + std::to_string(targets.cols()) +
                    " targets");
}

// A single column p becomes the two columns (1 - p, p).
Matrix asClasses(const Matrix &m)
{
    if (m.cols() != 1)
        return m;
    Matrix out(m.rows(), 2);
    for (std::size_t r = 0; r < m.rows(); r++) {
        out(r, 0) = 1 - m(r, 0);
        out(r, 1) = m(r, 0);
    }
    return out;
}

Scalar ratio(std::size_t num, std::size_t den)
{
    return den == 0 ? 0 : static_cast<Scalar>(num) / static_cast<Scalar>(den);
}

std::ofstream openOutput(const std::string &path)
{
    std::ofstream out(path);
    if (!out)
        throw Error("cannot write '" + path + "'");
    out.precision(17);
    return out;
}

void writeNumber(std::ostream &out, Scalar value)
{
    if (std::isfinite(value))
        out << value;
    else
        out << "null";
}

std::string escape(const std::string &text)
{
    std::string out;
    for (const char c : text) {
        if (c == '"' || c == '\\')
            out += '\\';
        out += c;
    }
    return out;
}

} // namespace

std::vector<std::size_t> classesOf(const Matrix &scores)
{
    std::vector<std::size_t> classes(scores.rows());

    for (std::size_t r = 0; r < scores.rows(); r++) {
        if (scores.cols() == 1) {
            classes[r] = scores(r, 0) >= 0.5 ? 1 : 0;
            continue;
        }
        std::size_t best = 0;
        for (std::size_t c = 1; c < scores.cols(); c++)
            if (scores(r, c) > scores(r, best))
                best = c;
        classes[r] = best;
    }
    return classes;
}

std::size_t ConfusionMatrix::classes() const noexcept
{
    return counts.size();
}

std::size_t ConfusionMatrix::total() const noexcept
{
    std::size_t sum = 0;
    for (const auto &row : counts)
        sum += std::accumulate(row.begin(), row.end(), std::size_t{0});
    return sum;
}

ConfusionMatrix confusion(const Matrix &probabilities, const Matrix &targets)
{
    requireSameShape("confusion", probabilities, targets);
    const std::size_t classes = std::max<std::size_t>(probabilities.cols(), 2);
    const auto predicted = classesOf(probabilities);
    const auto expected = classesOf(targets);
    ConfusionMatrix matrix{std::vector<std::vector<std::size_t>>(
        classes, std::vector<std::size_t>(classes, 0))};

    for (std::size_t r = 0; r < predicted.size(); r++)
        matrix.counts[expected[r]][predicted[r]]++;
    return matrix;
}

ConfusionMatrix confusion(const std::vector<std::size_t> &expected,
                          const std::vector<std::size_t> &predicted, std::size_t classes)
{
    if (expected.size() != predicted.size())
        throw Error("confusion: " + std::to_string(expected.size()) + " expected classes for " +
                    std::to_string(predicted.size()) + " predictions");
    if (expected.empty() || classes == 0)
        throw Error("confusion: empty input");
    ConfusionMatrix matrix{std::vector<std::vector<std::size_t>>(
        classes, std::vector<std::size_t>(classes, 0))};

    for (std::size_t r = 0; r < expected.size(); r++) {
        if (expected[r] >= classes || predicted[r] >= classes)
            throw Error("confusion: class index out of range at row " + std::to_string(r));
        matrix.counts[expected[r]][predicted[r]]++;
    }
    return matrix;
}

ConfusionMatrix merge(const ConfusionMatrix &confusion, const std::vector<std::size_t> &groupOf,
                      std::size_t groups)
{
    if (groupOf.size() != confusion.classes())
        throw Error("merge: " + std::to_string(groupOf.size()) + " groups given for " +
                    std::to_string(confusion.classes()) + " classes");
    ConfusionMatrix merged{std::vector<std::vector<std::size_t>>(
        groups, std::vector<std::size_t>(groups, 0))};

    for (std::size_t c = 0; c < groupOf.size(); c++)
        if (groupOf[c] >= groups)
            throw Error("merge: class " + std::to_string(c) + " goes to group " +
                        std::to_string(groupOf[c]) + ", but there are " + std::to_string(groups));
    for (std::size_t e = 0; e < groupOf.size(); e++)
        for (std::size_t p = 0; p < groupOf.size(); p++)
            merged.counts[groupOf[e]][groupOf[p]] += confusion.counts[e][p];
    return merged;
}

ClassificationReport report(const ConfusionMatrix &confusion)
{
    ClassificationReport rep;
    rep.confusion = confusion;
    const std::size_t classes = rep.confusion.classes();
    const std::size_t total = rep.confusion.total();
    std::size_t correct = 0;

    if (classes == 0 || total == 0)
        throw Error("report: empty confusion matrix");
    for (const auto &row : rep.confusion.counts)
        if (row.size() != classes)
            throw Error("report: the confusion matrix is not square");
    rep.perClass.resize(classes);
    for (std::size_t c = 0; c < classes; c++) {
        ClassMetrics &m = rep.perClass[c];
        const std::size_t hits = rep.confusion.counts[c][c];
        for (std::size_t other = 0; other < classes; other++) {
            m.support += rep.confusion.counts[c][other];
            m.predicted += rep.confusion.counts[other][c];
        }
        m.precision = ratio(hits, m.predicted);
        m.recall = ratio(hits, m.support);
        m.f1 = m.precision + m.recall == 0
                   ? 0
                   : 2 * m.precision * m.recall / (m.precision + m.recall);
        correct += hits;
        const Scalar weight = static_cast<Scalar>(m.support) / static_cast<Scalar>(total);
        rep.macroPrecision += m.precision / static_cast<Scalar>(classes);
        rep.macroRecall += m.recall / static_cast<Scalar>(classes);
        rep.macroF1 += m.f1 / static_cast<Scalar>(classes);
        rep.weightedPrecision += m.precision * weight;
        rep.weightedRecall += m.recall * weight;
        rep.weightedF1 += m.f1 * weight;
    }
    rep.accuracy = ratio(correct, total);
    return rep;
}

ClassificationReport report(const Matrix &probabilities, const Matrix &targets)
{
    ClassificationReport rep = report(confusion(probabilities, targets));

    const Matrix p = asClasses(probabilities);
    const auto expected = classesOf(targets);
    for (std::size_t r = 0; r < p.rows(); r++)
        rep.logLoss -= std::log(std::max(p(r, expected[r]), logFloor));
    rep.logLoss /= static_cast<Scalar>(p.rows());
    return rep;
}

Scalar topKAccuracy(const Matrix &probabilities, const Matrix &targets, std::size_t k)
{
    requireSameShape("top_k_accuracy", probabilities, targets);
    const Matrix p = asClasses(probabilities);
    if (k == 0 || k > p.cols())
        throw Error("top_k_accuracy: k must be between 1 and " + std::to_string(p.cols()));
    const auto expected = classesOf(targets);
    std::size_t hits = 0;

    for (std::size_t r = 0; r < p.rows(); r++) {
        // Rank of the expected class: classes strictly more probable, plus equal ones with a lower
        // index (the same tie rule as classesOf).
        const std::size_t e = expected[r];
        std::size_t rank = 0;
        for (std::size_t c = 0; c < p.cols(); c++)
            if (p(r, c) > p(r, e) || (p(r, c) == p(r, e) && c < e))
                rank++;
        hits += rank < k;
    }
    return ratio(hits, p.rows());
}

std::vector<Scalar> rocAuc(const Matrix &probabilities, const Matrix &targets)
{
    requireSameShape("roc_auc", probabilities, targets);
    const Matrix p = asClasses(probabilities);
    const auto expected = classesOf(targets);
    const std::size_t n = p.rows();
    std::vector<Scalar> auc(p.cols());
    std::vector<std::size_t> order(n);
    std::vector<Scalar> rank(n);

    for (std::size_t c = 0; c < p.cols(); c++) {
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](std::size_t a, std::size_t b) { return p(a, c) < p(b, c); });
        // Average rank (1-based) over each group of equal scores.
        for (std::size_t i = 0; i < n;) {
            std::size_t j = i;
            while (j + 1 < n && p(order[j + 1], c) == p(order[i], c))
                j++;
            const Scalar average = static_cast<Scalar>(i + j + 2) / 2;
            for (std::size_t k = i; k <= j; k++)
                rank[order[k]] = average;
            i = j + 1;
        }
        Scalar positiveRanks = 0;
        std::size_t positives = 0;
        for (std::size_t r = 0; r < n; r++)
            if (expected[r] == c) {
                positiveRanks += rank[r];
                positives++;
            }
        const std::size_t negatives = n - positives;
        if (positives == 0 || negatives == 0) {
            auc[c] = std::numeric_limits<Scalar>::quiet_NaN();
            continue;
        }
        const Scalar pos = static_cast<Scalar>(positives);
        auc[c] = (positiveRanks - pos * (pos + 1) / 2) / (pos * static_cast<Scalar>(negatives));
    }
    return auc;
}

Calibration calibration(const Matrix &probabilities, const Matrix &targets, std::size_t bins)
{
    requireSameShape("calibration", probabilities, targets);
    if (bins == 0)
        throw Error("calibration: bins must be positive");
    const Matrix p = asClasses(probabilities);
    const auto predicted = classesOf(p);
    const auto expected = classesOf(targets);
    Calibration cal;
    cal.bins.resize(bins);

    for (std::size_t r = 0; r < p.rows(); r++) {
        const Scalar confidence = p(r, predicted[r]);
        const Scalar scaled = std::max(confidence, 0.0) * static_cast<Scalar>(bins);
        const auto index = std::min(bins - 1, static_cast<std::size_t>(scaled));
        CalibrationBin &bin = cal.bins[index];
        bin.count++;
        bin.confidence += confidence;
        bin.accuracy += predicted[r] == expected[r];
    }
    for (CalibrationBin &bin : cal.bins) {
        if (bin.count == 0)
            continue;
        bin.confidence /= static_cast<Scalar>(bin.count);
        bin.accuracy /= static_cast<Scalar>(bin.count);
        cal.expectedCalibrationError += static_cast<Scalar>(bin.count) /
                                        static_cast<Scalar>(p.rows()) *
                                        std::abs(bin.accuracy - bin.confidence);
    }
    return cal;
}

void writeHistoryCsv(const History &history, const std::string &path)
{
    std::ofstream out = openOutput(path);

    out << "epoch,train_loss,validation_loss,penalty,train_accuracy,validation_accuracy\n";
    for (std::size_t e = 0; e < history.size(); e++) {
        const EpochStats &s = history[e];
        out << e << ',' << s.trainLoss << ',' << s.validationLoss << ',' << s.penalty << ','
            << s.trainAccuracy << ',' << s.validationAccuracy << '\n';
    }
}

void writeReportJson(const ClassificationReport &rep, const std::string &path,
                     const std::vector<std::string> &classNames)
{
    std::ofstream out = openOutput(path);
    const std::size_t classes = rep.perClass.size();
    const auto name = [&](std::size_t c) {
        return c < classNames.size() ? escape(classNames[c]) : std::to_string(c);
    };

    out << "{\n  \"accuracy\": ";
    writeNumber(out, rep.accuracy);
    out << ",\n  \"log_loss\": ";
    writeNumber(out, rep.logLoss);
    out << ",\n  \"macro\": {\"precision\": ";
    writeNumber(out, rep.macroPrecision);
    out << ", \"recall\": ";
    writeNumber(out, rep.macroRecall);
    out << ", \"f1\": ";
    writeNumber(out, rep.macroF1);
    out << "},\n  \"weighted\": {\"precision\": ";
    writeNumber(out, rep.weightedPrecision);
    out << ", \"recall\": ";
    writeNumber(out, rep.weightedRecall);
    out << ", \"f1\": ";
    writeNumber(out, rep.weightedF1);
    out << "},\n  \"classes\": [\n";
    for (std::size_t c = 0; c < classes; c++) {
        const ClassMetrics &m = rep.perClass[c];
        out << "    {\"name\": \"" << name(c) << "\", \"precision\": ";
        writeNumber(out, m.precision);
        out << ", \"recall\": ";
        writeNumber(out, m.recall);
        out << ", \"f1\": ";
        writeNumber(out, m.f1);
        out << ", \"support\": " << m.support << ", \"predicted\": " << m.predicted << '}'
            << (c + 1 < classes ? ",\n" : "\n");
    }
    out << "  ],\n  \"confusion\": [\n";
    for (std::size_t c = 0; c < classes; c++) {
        out << "    [";
        for (std::size_t p = 0; p < classes; p++)
            out << rep.confusion.counts[c][p] << (p + 1 < classes ? ", " : "");
        out << ']' << (c + 1 < classes ? ",\n" : "\n");
    }
    out << "  ]\n}\n";
}

} // namespace my_torch
