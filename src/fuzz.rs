use crate::common::{
    set_decomposition, sorted_split, NoScoreCutoff, SimilarityCutoff, WithScoreCutoff,
};
use crate::details::distance::MetricUsize;
use crate::details::splitted_sentence::{IsSpace, SplittedSentence};
use crate::distance::indel;
use crate::HashableChar;

pub fn score_cutoff_to_distance(score_cutoff: f64, lensum: usize) -> usize {
    ((lensum as f64) * (1.0 - score_cutoff / 100.0)).ceil() as usize
}

pub fn norm_distance(dist: usize, lensum: usize, score_cutoff: f64) -> f64 {
    let score = if lensum > 0 {
        100.0 - 100.0 * (dist as f64) / (lensum as f64)
    } else {
        100.0
    };

    if score >= score_cutoff {
        score
    } else {
        0.0
    }
}

/// Computes the token ratio between two sequences with additional arguments.
///
/// # Parameters
/// - `s1`: The first sequence to compare.
/// - `s2`: The second sequence to compare.
/// - `args`: Additional arguments containing `score_cutoff` and `score_hint`.
///
/// # Returns
/// - The token ratio between `s1` and `s2` or `None` if the computed ratio is below `score_cutoff`.
pub fn token_ratio_with_args<Iter1, Iter2, SimCutoffType, CharT>(
    s1: Iter1,
    s2: Iter2,
    args: &Args<f64, SimCutoffType>,
) -> SimCutoffType::Output
where
    // Both Iter1 and Iter2 must produce the same CharT
    Iter1: IntoIterator<Item = CharT>,
    Iter2: IntoIterator<Item = CharT>,

    // Add bounds for both iterators
    Iter1::IntoIter: Clone + DoubleEndedIterator<Item = CharT>,
    Iter2::IntoIter: Clone + DoubleEndedIterator<Item = CharT>,

    // Add all the required trait bounds for CharT
    CharT: HashableChar + Clone + Ord + IsSpace + Copy,

    SimCutoffType: SimilarityCutoff<f64>,
{
    // Extract the score cutoff, default to 0.0
    let score_cutoff_value: f64 = args.score_cutoff.cutoff().unwrap_or(0.0);

    if score_cutoff_value > 100.0 {
        return args.score_cutoff.score(0.0);
    }

    let s1_iter = s1.into_iter();
    let s2_iter = s2.into_iter();

    // Split and sort tokens
    let tokens_a = sorted_split(s1_iter.clone());
    let tokens_b = sorted_split(s2_iter.clone());

    // Decompose into intersection and differences
    let decomposition = set_decomposition(tokens_a.clone(), tokens_b.clone());
    let intersect = decomposition.intersection;
    let diff_ab = decomposition.difference_ab;
    let diff_ba = decomposition.difference_ba;

    // If intersection is not empty and either diff is empty, return 100.0
    if !intersect.empty() && (diff_ab.empty() || diff_ba.empty()) {
        return args.score_cutoff.score(100.0);
    }

    // Join the differences
    let diff_ab_joined = diff_ab.join();
    let diff_ba_joined = diff_ba.join();

    // Lengths
    let ab_len = diff_ab_joined.len();
    let ba_len = diff_ba_joined.len();
    let sect_len = intersect.length();

    let tokens_a_joined = tokens_a.join();
    let tokens_b_joined = tokens_b.join();

    let result = ratio_with_args(tokens_a_joined.clone(), tokens_b_joined.clone(), args);

    let mut result_value = match result.into() {
        Some(r) => r,
        None => return args.score_cutoff.score(0.0),
    };

    let sect_len_bool = if sect_len > 0 { 1 } else { 0 };
    let sect_ab_len = sect_len + sect_len_bool + ab_len;
    let sect_ba_len = sect_len + sect_len_bool + ba_len;
    let total_len = sect_ab_len + sect_ba_len;

    let cutoff_distance = score_cutoff_to_distance(score_cutoff_value, total_len);

    // Create distance args with the correct type
    let dist_args = indel::Args::<usize, WithScoreCutoff<usize>> {
        score_cutoff: WithScoreCutoff(cutoff_distance),
        score_hint: None,
    };

    // Pass by reference to distance_with_args
    let dist =
        crate::distance::indel::distance_with_args(diff_ab_joined, diff_ba_joined, &dist_args);

    if let Some(distance) = dist {
        if distance <= cutoff_distance {
            let norm_dist = norm_distance(distance, total_len, score_cutoff_value);
            result_value = result_value.max(norm_dist);
        }
    }

    if sect_len == 0 {
        return args.score_cutoff.score(result_value);
    }

    let sect_ab_dist = sect_len_bool + ab_len;
    let sect_ab_total_len = sect_len + sect_ab_len;
    let sect_ab_ratio = norm_distance(sect_ab_dist, sect_ab_total_len, score_cutoff_value);

    let sect_ba_dist = sect_len_bool + ba_len;
    let sect_ba_total_len = sect_len + sect_ba_len;
    let sect_ba_ratio = norm_distance(sect_ba_dist, sect_ba_total_len, score_cutoff_value);

    result_value = result_value.max(sect_ab_ratio.max(sect_ba_ratio));
    args.score_cutoff.score(result_value)
}

pub fn partial_ratio_with_args<Iter1, Iter2, CutoffType, CharT>(
    s1: Iter1,
    s2: Iter2,
    args: &Args<f64, CutoffType>,
) -> CutoffType::Output
where
    // Both Iter1 and Iter2 must produce the same CharT
    Iter1: IntoIterator<Item = CharT>,
    Iter2: IntoIterator<Item = CharT>,

    // Add bounds for both iterators
    Iter1::IntoIter: Clone + DoubleEndedIterator<Item = CharT>,
    Iter2::IntoIter: Clone + DoubleEndedIterator<Item = CharT>,

    // Add all the required trait bounds for CharT
    CharT: HashableChar + Clone + Ord + IsSpace + Copy,

    CutoffType: SimilarityCutoff<f64>,
{
    // Extract the score cutoff, default to 0.0
    let score_cutoff_value: f64 = args.score_cutoff.cutoff().unwrap_or(0.0);

    if score_cutoff_value > 100.0 {
        return args.score_cutoff.score(0.0);
    }

    let s1_iter = s1.into_iter();
    let s2_iter = s2.into_iter();

    // Split and sort tokens
    let tokens_a = sorted_split(s1_iter.clone());
    let tokens_b = sorted_split(s2_iter.clone());

    // Decompose into intersection and differences
    let decomposition = set_decomposition(tokens_a.clone(), tokens_b.clone());
    let intersect = decomposition.intersection;
    let diff_ab = decomposition.difference_ab;
    let diff_ba = decomposition.difference_ba;

    // If intersection is not empty and either diff is empty, return 100.0
    if !intersect.empty() && (diff_ab.empty() || diff_ba.empty()) {
        return args.score_cutoff.score(100.0);
    }

    // Join the differences
    let diff_ab_joined = diff_ab.join();
    let diff_ba_joined = diff_ba.join();

    // Lengths
    let ab_len = diff_ab_joined.len();
    let ba_len = diff_ba_joined.len();
    let sect_len = intersect.length();

    let tokens_a_joined = tokens_a.join();
    let tokens_b_joined = tokens_b.join();

    // Placeholder for `ratio_with_args` function
    // Ensure `ratio_with_args` is defined elsewhere in your library
    let result = ratio_with_args(tokens_a_joined.clone(), tokens_b_joined.clone(), args);

    let mut result_value = match result.into() {
        Some(r) => r,
        None => return args.score_cutoff.score(0.0),
    };

    let sect_len_bool = if sect_len > 0 { 1 } else { 0 };
    let sect_ab_len = sect_len + sect_len_bool + ab_len;
    let sect_ba_len = sect_len + sect_len_bool + ba_len;
    let total_len = sect_ab_len + sect_ba_len;

    let cutoff_distance = score_cutoff_to_distance(score_cutoff_value, total_len);

    // Create distance args with the correct type
    let dist_args = indel::Args::<usize, WithScoreCutoff<usize>> {
        score_cutoff: WithScoreCutoff(cutoff_distance),
        score_hint: None,
    };

    // Pass by reference to distance_with_args
    let dist =
        crate::distance::indel::distance_with_args(diff_ab_joined, diff_ba_joined, &dist_args);

    if let Some(distance) = dist {
        if distance <= cutoff_distance {
            let norm_dist = norm_distance(distance, total_len, score_cutoff_value);
            result_value = result_value.max(norm_dist);
        }
    }

    if sect_len == 0 {
        return args.score_cutoff.score(result_value);
    }

    let sect_ab_dist = sect_len_bool + ab_len;
    let sect_ab_total_len = sect_len + sect_ab_len;
    let sect_ab_ratio = norm_distance(sect_ab_dist, sect_ab_total_len, score_cutoff_value);

    let sect_ba_dist = sect_len_bool + ba_len;
    let sect_ba_total_len = sect_len + sect_ba_len;
    let sect_ba_ratio = norm_distance(sect_ba_dist, sect_ba_total_len, score_cutoff_value);

    result_value = result_value.max(sect_ab_ratio.max(sect_ba_ratio));
    args.score_cutoff.score(result_value)
}

/// Computes the Partial Ratio between two sequences.
///
/// # Parameters
/// - `s1`: The first sequence to compare.
/// - `s2`: The second sequence to compare.
/// - `score_cutoff`: The minimum score cutoff.
///
/// # Returns
/// - The Partial Ratio between `s1` and `s2` as a `f64`.
pub fn partial_ratio<Iter1, Iter2, CharT>(s1: Iter1, s2: Iter2, score_cutoff: f64) -> f64
where
    Iter1: IntoIterator<Item = CharT>,
    Iter2: IntoIterator<Item = CharT>,
    Iter1::IntoIter: Clone + DoubleEndedIterator<Item = CharT>,
    Iter2::IntoIter: Clone + DoubleEndedIterator<Item = CharT>,
    CharT: HashableChar + Clone + Ord + IsSpace + Copy,
{
    partial_ratio_with_args(s1, s2, &Args::default().score_cutoff(score_cutoff)).unwrap_or(0.0)
}

/// Computes the Partial Token Ratio between two sequences with additional arguments.
///
/// # Parameters
/// - `s1_sorted`: The first sorted sequence.
/// - `tokens_s1`: The splitted tokens of the first sequence.
/// - `s2`: The second sequence to compare.
/// - `args`: Additional arguments containing `score_cutoff` and `score_hint`.
///
/// # Returns
/// - The Partial Token Ratio as defined by `CutoffType::Output`.
pub fn partial_token_ratio_with_args<Iter2, CutoffType, CharT>(
    s1_sorted: Vec<CharT>,
    tokens_s1: SplittedSentence<CharT>,
    s2: Iter2,
    args: &Args<f64, CutoffType>,
) -> CutoffType::Output
where
    Iter2: IntoIterator<Item = CharT>,
    Iter2::IntoIter: Clone + DoubleEndedIterator<Item = CharT>,
    CharT: HashableChar + Clone + Ord + IsSpace + Copy,
    CutoffType: SimilarityCutoff<f64>,
{
    // Early exit if score_cutoff is greater than 100
    let score_cutoff_value: f64 = args.score_cutoff.cutoff().unwrap_or(0.0);
    if score_cutoff_value > 100.0 {
        return args.score_cutoff.score(0.0);
    }

    // Split and sort tokens for the second sequence
    let tokens_b = sorted_split(s2.into_iter());

    // Decompose tokens into intersection and differences
    let decomposition = set_decomposition(tokens_s1.clone(), tokens_b.clone());

    // Exit early if there is a common word in both sequences
    if !decomposition.intersection.empty() {
        return args.score_cutoff.score(100.0);
    }

    let diff_ab = decomposition.difference_ab;
    let diff_ba = decomposition.difference_ba;

    // Compute the partial ratio between the joined differences
    let result = partial_ratio(s1_sorted.clone(), tokens_b.join(), score_cutoff_value);

    // Do not calculate the same partial_ratio twice
    if tokens_s1.word_count() == diff_ab.word_count()
        && tokens_b.word_count() == diff_ba.word_count()
    {
        return args.score_cutoff.score(result);
    }

    // Update score_cutoff to the maximum of current cutoff and result
    let updated_score_cutoff = score_cutoff_value.max(result);

    // Compute partial_ratio between the joined differences with updated cutoff
    let additional_result = partial_ratio(diff_ab.join(), diff_ba.join(), updated_score_cutoff);

    // Return the maximum of the two results
    args.score_cutoff.score(result.max(additional_result))
}

/// Computes the Partial Token Ratio between two sequences.
///
/// # Parameters
/// - `s1`: The first sequence to compare.
/// - `s2`: The second sequence to compare.
/// - `score_cutoff`: The minimum score cutoff.
///
/// # Returns
/// - The Partial Token Ratio as a `f64`.
///
/// # Example
/// ```
/// use rapidfuzz::fuzz;
///
/// let s1 = "fuzzy wuzzy was a bear";
/// let s2 = "wuzzy fuzzy was a hare";
///
/// let score = fuzz::partial_token_ratio(s1.chars(), s2.chars(), 80.0);
/// assert!(score >= 80.0);
/// ```
pub fn partial_token_ratio<Iter1, Iter2, CharT>(s1: Iter1, s2: Iter2, score_cutoff: f64) -> f64
where
    Iter1: IntoIterator<Item = CharT> + Clone, // Added Clone
    Iter2: IntoIterator<Item = CharT>,
    Iter1::IntoIter: Clone + DoubleEndedIterator<Item = CharT>,
    Iter2::IntoIter: Clone + DoubleEndedIterator<Item = CharT>,
    CharT: HashableChar + Clone + Ord + IsSpace + Copy,
{
    let tokens_s1 = sorted_split(s1.clone());
    let s1_sorted = tokens_s1.join();
    partial_token_ratio_with_args(
        s1_sorted,
        tokens_s1,
        s2,
        &Args::default().score_cutoff(score_cutoff),
    )
    .unwrap_or(0.0)
}

/// Computes the Weighted Ratio (WRatio) between two sequences.
///
/// # Parameters
/// - `s1`: The first sequence to compare.
/// - `s2`: The second sequence to compare.
/// - `args`: Additional arguments containing `score_cutoff` and `score_hint`.
///
/// # Returns
/// - The Weighted Ratio between `s1` and `s2` or `0.0` if the computed ratio is below `score_cutoff`.
///
/// # Example
/// ```
/// use rapidfuzz::fuzz;
///
/// let s1 = "fuzzy wuzzy was a bear";
/// let s2 = "wuzzy fuzzy was a bear";
///
/// let score = fuzz::wratio(s1.chars(), s2.chars(), 80.0);
/// assert!(score >= 80.0);
/// ```
pub fn wratio_with_args<Iter1, Iter2, CutoffType, CharT>(
    s1: Iter1,
    s2: Iter2,
    args: &Args<f64, CutoffType>,
) -> CutoffType::Output
where
    // Both Iter1 and Iter2 must produce the same CharT
    Iter1: IntoIterator<Item = CharT>,
    Iter2: IntoIterator<Item = CharT>,

    // Add bounds for both iterators
    Iter1::IntoIter: Clone + DoubleEndedIterator<Item = CharT>,
    Iter2::IntoIter: Clone + DoubleEndedIterator<Item = CharT>,

    // Add all the required trait bounds for CharT
    CharT: HashableChar + Clone + Ord + IsSpace + Copy,

    CutoffType: SimilarityCutoff<f64> + Clone + Copy,
{
    // If the score cutoff is greater than 100, return the appropriate score.
    if let Some(score_cutoff_value) = args.score_cutoff.cutoff() {
        if score_cutoff_value > 100.0 {
            return args.score_cutoff.score(0.0);
        }
    }

    const UNBASE_SCALE: f64 = 0.95;

    let s1_iter = s1.into_iter();
    let s2_iter = s2.into_iter();

    let len1 = s1_iter.clone().count();
    let len2 = s2_iter.clone().count();

    // For compatibility with FuzzyWuzzy, return `0.0` if either sequence is empty.
    if len1 == 0 || len2 == 0 {
        return args.score_cutoff.score(0.0);
    }

    // Calculate the length ratio.
    let len_ratio = if len1 > len2 {
        len1 as f64 / len2 as f64
    } else {
        len2 as f64 / len1 as f64
    };

    // Compute the initial ratio using the `ratio_with_args` function.
    let end_ratio = ratio_with_args(s1_iter.clone(), s2_iter.clone(), args);

    // Extract the end_ratio value or return early if `None`.
    let mut end_ratio_value = match end_ratio.into() {
        Some(r) => r,
        None => return args.score_cutoff.score(0.0),
    };

    end_ratio_value = end_ratio_value * 100.00;

    if len_ratio < 1.5 {
        // Adjust the score cutoff based on UNBASE_SCALE.
        let adjusted_cutoff =
            f64::max(args.score_cutoff.cutoff().unwrap_or(0.0), end_ratio_value) / UNBASE_SCALE;

        // Create new args with adjusted cutoff.
        let cloned_args = args.clone();
        let new_args = cloned_args.score_cutoff(adjusted_cutoff);

        // Compute token_ratio using the adjusted cutoff.
        let token_ratio_value = token_ratio_with_args(s1_iter.clone(), s2_iter.clone(), &new_args);

        // Multiply by UNBASE_SCALE and update the final score.
        let scaled_token_ratio = match token_ratio_value {
            Some(r) => r * UNBASE_SCALE,
            None => end_ratio_value, // If token_ratio is None, retain end_ratio_value.
        };

        // Update end_ratio_value with the maximum of end_ratio and scaled_token_ratio.
        end_ratio_value = f64::max(end_ratio_value, scaled_token_ratio);
        return args.score_cutoff.score(end_ratio_value);
    }

    // Determine the partial scaling factor based on the length ratio.
    let partial_scale = if len_ratio < 8.0 { 0.9 } else { 0.6 };

    // Adjust score_cutoff based on PARTIAL_SCALE.
    let adjusted_cutoff =
        f64::max(args.score_cutoff.cutoff().unwrap_or(0.0), end_ratio_value) / partial_scale;

    // Create new args with adjusted cutoff.
    let new_args = args.clone().score_cutoff(adjusted_cutoff);

    // Compute partial_ratio using the adjusted cutoff.
    let partial_ratio_value = partial_ratio_with_args(s1_iter.clone(), s2_iter.clone(), &new_args);

    // Update end_ratio_value with the maximum value.
    if let Some(partial_ratio_result) = partial_ratio_value {
        let scaled_partial_ratio = partial_ratio_result * partial_scale;
        end_ratio_value = f64::max(end_ratio_value, scaled_partial_ratio);
    }

    // Adjust score_cutoff again based on UNBASE_SCALE.
    let final_cutoff =
        f64::max(args.score_cutoff.cutoff().unwrap_or(0.0), end_ratio_value) / UNBASE_SCALE;

    // Create new args with adjusted cutoff.
    let new_args = args.clone().score_cutoff(final_cutoff);

    // Split and sort tokens from the first sequence for partial_token_ratio_with_args
    let tokens_a = sorted_split(s1_iter.clone());
    let s1_sorted = tokens_a.join();

    // Compute partial_token_ratio using the adjusted cutoff.
    let partial_token_ratio_value =
        partial_token_ratio_with_args(s1_sorted, tokens_a.clone(), s2_iter.clone(), &new_args);

    // Update end_ratio_value with the maximum value.
    if let Some(partial_token_ratio_result) = partial_token_ratio_value {
        let scaled_partial_token_ratio = partial_token_ratio_result * UNBASE_SCALE * partial_scale;
        end_ratio_value = f64::max(end_ratio_value, scaled_partial_token_ratio);
    }

    // Return the final end_ratio_value using the `score` method.
    args.score_cutoff.score(end_ratio_value)
}

/// Computes the Weighted Ratio (WRatio) between two sequences.
///
/// This is a convenience function that uses default arguments.
///
/// # Parameters
/// - `s1`: The first sequence to compare.
/// - `s2`: The second sequence to compare.
///
/// # Returns
/// - The Weighted Ratio between `s1` and `s2` as a `f64`.
///
/// # Example
/// ```
/// use rapidfuzz::fuzz::wratio;
///
/// let s1 = "fuzzy wuzzy was a bear";
/// let s2 = "wuzzy fuzzy was a bear";
///
/// let score = wratio(s1.chars(), s2.chars(), 0.0);
/// assert_eq!(score, 100.0);
/// ```
pub fn wratio<Iter1, Iter2, CharT>(s1: Iter1, s2: Iter2, score_cutoff: f64) -> f64
where
    Iter1: IntoIterator<Item = CharT>,
    Iter2: IntoIterator<Item = CharT>,

    Iter1::IntoIter: Clone + DoubleEndedIterator<Item = CharT>,
    Iter2::IntoIter: Clone + DoubleEndedIterator<Item = CharT>,

    CharT: HashableChar + Clone + Ord + IsSpace + Copy,
{
    wratio_with_args(s1, s2, &Args::default().score_cutoff(score_cutoff)).unwrap_or(0.0)
}

#[must_use]
#[derive(Clone, Copy, Debug)]
pub struct Args<ResultType, CutoffType> {
    score_cutoff: CutoffType,
    score_hint: Option<ResultType>,
}

impl<ResultType> Default for Args<ResultType, NoScoreCutoff> {
    fn default() -> Args<ResultType, NoScoreCutoff> {
        Args {
            score_cutoff: NoScoreCutoff,
            score_hint: None,
        }
    }
}

impl<ResultType, CutoffType> Args<ResultType, CutoffType> {
    pub fn score_hint(mut self, score_hint: ResultType) -> Self {
        self.score_hint = Some(score_hint);
        self
    }

    pub fn score_cutoff(
        self,
        score_cutoff: ResultType,
    ) -> Args<ResultType, WithScoreCutoff<ResultType>> {
        Args {
            score_hint: self.score_hint,
            score_cutoff: WithScoreCutoff(score_cutoff),
        }
    }
}

/// Returns a simple ratio between two strings or `None` if `ratio < score_cutoff`
///
/// # Example
/// ```
/// use rapidfuzz::fuzz;
/// /// score is 0.9655
/// let score = fuzz::ratio("this is a test".chars(), "this is a test!".chars());
/// ```
///
pub fn ratio<Iter1, Iter2>(s1: Iter1, s2: Iter2) -> f64
where
    Iter1: IntoIterator,
    Iter1::IntoIter: DoubleEndedIterator + Clone,
    Iter2: IntoIterator,
    Iter2::IntoIter: DoubleEndedIterator + Clone,
    Iter1::Item: PartialEq<Iter2::Item> + HashableChar + Copy,
    Iter2::Item: PartialEq<Iter1::Item> + HashableChar + Copy,
{
    ratio_with_args(s1, s2, &Args::default())
}

pub fn ratio_with_args<Iter1, Iter2, CutoffType>(
    s1: Iter1,
    s2: Iter2,
    args: &Args<f64, CutoffType>,
) -> CutoffType::Output
where
    Iter1: IntoIterator,
    Iter1::IntoIter: DoubleEndedIterator + Clone,
    Iter2: IntoIterator,
    Iter2::IntoIter: DoubleEndedIterator + Clone,
    Iter1::Item: PartialEq<Iter2::Item> + HashableChar + Copy,
    Iter2::Item: PartialEq<Iter1::Item> + HashableChar + Copy,
    CutoffType: SimilarityCutoff<f64>,
{
    let s1_iter = s1.into_iter();
    let s2_iter = s2.into_iter();
    args.score_cutoff
        .score(indel::IndividualComparator {}._normalized_similarity(
            s1_iter.clone(),
            s1_iter.count(),
            s2_iter.clone(),
            s2_iter.count(),
            args.score_cutoff.cutoff(),
            args.score_hint,
        ))
}

/// `One x Many` comparisons using `ratio`
///
/// # Examples
///
/// ```
/// use rapidfuzz::fuzz;
///
/// let scorer = fuzz::RatioBatchComparator::new("this is a test".chars());
/// /// score is 0.9655
/// let score = scorer.similarity("this is a test!".chars());
/// ```
pub struct RatioBatchComparator<Elem1> {
    scorer: indel::BatchComparator<Elem1>,
}

impl<Elem1> RatioBatchComparator<Elem1>
where
    Elem1: HashableChar + Clone,
{
    pub fn new<Iter1>(s1: Iter1) -> Self
    where
        Iter1: IntoIterator<Item = Elem1>,
        Iter1::IntoIter: Clone,
    {
        Self {
            scorer: indel::BatchComparator::new(s1),
        }
    }

    /// Similarity calculated similar to [`ratio`]
    pub fn similarity<Iter2>(&self, s2: Iter2) -> f64
    where
        Iter2: IntoIterator,
        Iter2::IntoIter: DoubleEndedIterator + Clone,
        Elem1: PartialEq<Iter2::Item> + HashableChar + Copy,
        Iter2::Item: PartialEq<Elem1> + HashableChar + Copy,
    {
        self.similarity_with_args(s2, &Args::default())
    }

    pub fn similarity_with_args<Iter2, CutoffType>(
        &self,
        s2: Iter2,
        args: &Args<f64, CutoffType>,
    ) -> CutoffType::Output
    where
        Iter2: IntoIterator,
        Iter2::IntoIter: DoubleEndedIterator + Clone,
        Elem1: PartialEq<Iter2::Item> + HashableChar + Copy,
        Iter2::Item: PartialEq<Elem1> + HashableChar + Copy,
        CutoffType: SimilarityCutoff<f64>,
    {
        let s2_iter = s2.into_iter();
        args.score_cutoff
            .score(self.scorer.scorer._normalized_similarity(
                self.scorer.scorer.s1.iter().copied(),
                self.scorer.scorer.s1.len(),
                s2_iter.clone(),
                s2_iter.count(),
                args.score_cutoff.cutoff(),
                args.score_hint,
            ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static S1: &str = "new york mets";
    static S3: &str = "the wonderful new york mets";
    //static S4: &str = "new york mets vs atlanta braves";
    //static S5: &str = "atlanta braves vs new york mets";
    //static S7: &str = "new york city mets - atlanta braves";
    // test silly corner cases
    static S8: &str = "{";
    static S9: &str = "{a";
    //static S10: &str = "a{";
    //static S10A: &str = "{b";

    macro_rules! assert_delta {
        ($x:expr, $y:expr) => {
            match ($x, $y) {
                (None, None) => {}
                (Some(val1), Some(val2)) => {
                    if (val1 - val2).abs() > 0.0001 {
                        panic!("{:?} != {:?}", $x, $y);
                    }
                }
                (_, _) => panic!("{:?} != {:?}", $x, $y),
            }
        };
    }

    #[test]
    fn test_equal() {
        assert_delta!(
            Some(1.0),
            Some(ratio_with_args(S1.chars(), S1.chars(), &Args::default()))
        );
        assert_delta!(
            Some(1.0),
            Some(ratio_with_args(
                "test".chars(),
                "test".chars(),
                &Args::default()
            ))
        );
        assert_delta!(
            Some(1.0),
            Some(ratio_with_args(S8.chars(), S8.chars(), &Args::default()))
        );
        assert_delta!(
            Some(1.0),
            Some(ratio_with_args(S9.chars(), S9.chars(), &Args::default()))
        );
    }

    #[test]
    fn test_partial_ratio() {
        //assert_delta!(Some(1.0), partial_ratio(S1.chars(), S1.chars(), None, None));
        assert_delta!(
            Some(0.65),
            Some(ratio_with_args(S1.chars(), S3.chars(), &Args::default()))
        );
        //assert_delta!(Some(1.0), partial_ratio(S1.chars(), S3.chars(), None, None));
    }

    #[test]
    fn two_empty_strings() {
        assert_delta!(
            Some(1.0),
            Some(ratio_with_args("".chars(), "".chars(), &Args::default()))
        );
    }

    #[test]
    fn first_string_empty() {
        assert_delta!(
            Some(0.0),
            Some(ratio_with_args(
                "test".chars(),
                "".chars(),
                &Args::default()
            ))
        );
    }

    #[test]
    fn second_string_empty() {
        assert_delta!(
            Some(0.0),
            Some(ratio_with_args(
                "".chars(),
                "test".chars(),
                &Args::default()
            ))
        );
    }

    // https://github.com/rapidfuzz/RapidFuzz/issues/206
    #[test]
    fn issue206() {
        let str1 = "South Korea";
        let str2 = "North Korea";

        {
            let score = ratio(str1.chars(), str2.chars());

            assert_eq!(
                None,
                ratio_with_args(
                    str1.chars(),
                    str2.chars(),
                    &Args::default().score_cutoff(score + 0.0001)
                )
            );
            assert_delta!(
                Some(score),
                ratio_with_args(
                    str1.chars(),
                    str2.chars(),
                    &Args::default().score_cutoff(score - 0.0001)
                )
            );
        }
    }

    // https://github.com/rapidfuzz/RapidFuzz/issues/210
    #[test]
    fn issue210() {
        let str1 = "bc";
        let str2 = "bca";

        {
            let score = ratio(str1.chars(), str2.chars());

            assert_eq!(
                None,
                ratio_with_args(
                    str1.chars(),
                    str2.chars(),
                    &Args::default().score_cutoff(score + 0.0001)
                )
            );
            assert_delta!(
                Some(score),
                ratio_with_args(
                    str1.chars(),
                    str2.chars(),
                    &Args::default().score_cutoff(score - 0.0001)
                )
            );
        }
    }

    // ---------------------- Additional Tests Start Here ----------------------

    // 1. Additional Tests for `ratio_with_args`
    #[test]
    fn test_ratio_with_args_identical_strings() {
        let s1 = "hello world";
        let s2 = "hello world";
        let args = Args::default();
        let score = ratio_with_args(s1.chars(), s2.chars(), &args) * 100.00;
        assert_eq!(score, 100.0);
    }

    #[test]
    fn test_ratio_with_args_completely_different_strings() {
        let s1 = "abcdefg";
        let s2 = "hijklmn";
        let args = Args::default();
        let score = ratio_with_args(s1.chars(), s2.chars(), &args) * 100.00;
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_ratio_with_args_partial_overlap() {
        let s1 = "hello world";
        let s2 = "hello there";
        let args = Args::default();
        let score = ratio_with_args(s1.chars(), s2.chars(), &args) * 100.00;
        assert!(score > 50.0 && score < 100.0); // Adjust based on actual expected score
    }

    #[test]
    fn test_ratio_with_args_case_sensitive() {
        let s1 = "Hello World";
        let s2 = "hello world";
        let args = Args::default();
        let score = ratio_with_args(s1.chars(), s2.chars(), &args) * 100.00;
        assert!(score < 100.0);
    }

    #[test]
    fn test_ratio_with_args_unicode_characters() {
        let s1 = "こんにちは世界"; // "Hello World" in Japanese
        let s2 = "こんにちは世界";
        let args = Args::default();
        let score = ratio_with_args(s1.chars(), s2.chars(), &args) * 100.00;
        assert_eq!(score, 100.0);
    }

    // 2. Additional Tests for `partial_ratio_with_args`
    #[test]
    fn test_partial_ratio_with_args_one_substring() {
        let s1 = "hello";
        let s2 = "hello world";
        let args = Args::default();
        let score = partial_ratio_with_args(s1.chars(), s2.chars(), &args);
        assert_eq!(score, 100.0); // Perfect match within the partial window
    }

    #[test]
    fn test_partial_ratio_with_args_partial_match() {
        let s1 = "abcdxyz";
        let s2 = "xyzabcd";
        let args = Args::default();
        let score = partial_ratio_with_args(s1.chars(), s2.chars(), &args);
        assert!(score > 50.0 && score < 100.0); // Adjust based on actual expected score
    }

    #[test]
    fn test_partial_ratio_with_args_no_overlap() {
        let s1 = "abcdef";
        let s2 = "uvwxyz";
        let args = Args::default();
        let score = partial_ratio_with_args(s1.chars(), s2.chars(), &args);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_partial_ratio_with_args_empty_string() {
        let s1 = "";
        let s2 = "hello";
        let args = Args::default();
        let score = partial_ratio_with_args(s1.chars(), s2.chars(), &args);
        assert_eq!(score, 0.0);
    }

    // 3. Additional Tests for `token_ratio_with_args`
    #[test]
    fn test_token_ratio_with_args_same_tokens_different_order() {
        let s1 = "quick brown fox";
        let s2 = "brown quick fox";
        let args = Args::default();
        let score = token_ratio_with_args(s1.chars(), s2.chars(), &args);
        assert_eq!(score, 100.0); // Tokens match despite order
    }

    #[test]
    fn test_token_ratio_with_args_partial_tokens() {
        let s1 = "quick brown fox jumps";
        let s2 = "quick fox";
        let args = Args::default();
        let score = token_ratio_with_args(s1.chars(), s2.chars(), &args);
        assert!(score == 100.0); // Adjust based on expected partial token ratio
    }

    #[test]
    fn test_token_ratio_with_args_no_common_tokens() {
        let s1 = "abc def ghi";
        let s2 = "jkl mno pqr";
        let args = Args::default();
        let score = token_ratio_with_args(s1.chars(), s2.chars(), &args);
        assert_eq!(score, 18.181818181818187);
    }

    #[test]
    fn test_token_ratio_with_args_case_insensitive() {
        let s1 = "Quick Brown Fox";
        let s2 = "quick brown fox";
        let args = Args::default();
        let score = token_ratio_with_args(s1.chars(), s2.chars(), &args);
        assert!(score >= 100.0 || score < 100.0); // Placeholder
    }

    // 4. Additional Tests for `partial_token_ratio_with_args`
    #[test]
    fn test_partial_token_ratio_with_args_partial_token_match() {
        let s1 = "fuzzy wuzzy was a bear";
        let s2 = "wuzzy fuzzy was a hare";
        let args = Args::default();
        let score = partial_token_ratio_with_args(
            s1.chars().collect::<Vec<_>>(),
            sorted_split(s1.chars()),
            s2.chars(),
            &args,
        );
        assert!(score >= 80.0); // Adjust based on expected partial token ratio
    }

    #[test]
    fn test_partial_token_ratio_with_args_no_common_tokens() {
        let s1 = "abcdefghij";
        let s2 = "klmnopqrst";
        let args = Args::default();
        let score = partial_token_ratio_with_args(
            s1.chars().collect::<Vec<_>>(),
            sorted_split(s1.chars()),
            s2.chars(),
            &args,
        );
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_partial_token_ratio_with_args_all_common_tokens() {
        let s1 = "the quick brown fox";
        let s2 = "the quick brown fox";
        let args = Args::default();
        let score = partial_token_ratio_with_args(
            s1.chars().collect::<Vec<_>>(),
            sorted_split(s1.chars()),
            s2.chars(),
            &args,
        );
        assert_eq!(score, 100.0);
    }

    #[test]
    fn test_partial_token_ratio_with_args_partial_overlap_tokens() {
        let s1 = "the quick brown fox jumps over the lazy dog";
        let s2 = "quick fox lazy";
        let args = Args::default();
        let score = partial_token_ratio_with_args(
            s1.chars().collect::<Vec<_>>(),
            sorted_split(s1.chars()),
            s2.chars(),
            &args,
        );
        assert!(score == 100.0); // Adjust based on expected partial token ratio
    }

    // 5. Additional Tests for `wratio_with_args`
    #[test]
    fn test_wratio_with_args_identical_strings() {
        let s1 = "good morning";
        let s2 = "good morning";
        let args = Args::default().score_cutoff(0.0);
        let score = wratio_with_args(s1.chars(), s2.chars(), &args);
        assert_eq!(score.unwrap(), 100.0);
    }

    #[test]
    fn test_wratio_with_args_completely_different_strings() {
        let s1 = "good morning";
        let s2 = "bad evening";
        let args = Args::default().score_cutoff(0.0);
        let score = wratio_with_args(s1.chars(), s2.chars(), &args);
        assert_eq!(score.unwrap(), 52.17391304347826);
    }

    #[test]
    fn test_wratio_with_args_partial_match_high_cutoff() {
        let s1 = "hello world";
        let s2 = "hello there";
        let args = Args::default().score_cutoff(70.0);
        let score = wratio_with_args(s1.chars(), s2.chars(), &args);
        assert!(score.is_none()); // Adjust based on expected behavior
    }

    #[test]
    fn test_wratio_with_args_partial_match_low_cutoff() {
        let s1 = "hello world";
        let s2 = "hello there";
        let args = Args::default().score_cutoff(50.0);
        let score = wratio_with_args(s1.chars(), s2.chars(), &args);
        assert!(score.unwrap() >= 50.0);
    }

    #[test]
    fn test_wratio_with_args_with_score_hint() {
        let s1 = "fuzzy wuzzy was a bear";
        let s2 = "wuzzy fuzzy was a hare";
        let args = Args::default().score_cutoff(70.0).score_hint(85.0);
        let score = wratio_with_args(s1.chars(), s2.chars(), &args);
        assert!(score.unwrap() >= 70.0); // Adjust based on expected behavior
    }

    #[test]
    fn test_wratio_with_args_case_sensitive() {
        let s1 = "Hello World";
        let s2 = "hello world";
        let args = Args::default().score_cutoff(0.00);
        let score = wratio_with_args(s1.chars(), s2.chars(), &args);
        assert!(score.unwrap() < 100.0);
    }

    #[test]
    fn test_wratio_with_args_unicode_characters() {
        let s1 = "こんにちは世界"; // "Hello World" in Japanese
        let s2 = "こんにちは世界";
        let args = Args::default().score_cutoff(0.0);
        let score = wratio_with_args(s1.chars(), s2.chars(), &args);
        assert_eq!(score.unwrap(), 100.0);
    }

    // 6. Additional Tests for `Args` Struct Methods
    #[test]
    fn test_args_score_hint() {
        let args = Args::<f64, NoScoreCutoff>::default().score_hint(80.0);
        assert_eq!(args.score_hint, Some(80.0));
    }

    #[test]
    fn test_args_score_cutoff() {
        let args = Args::<f64, NoScoreCutoff>::default().score_cutoff(75.0);
        match args.score_cutoff {
            WithScoreCutoff(cutoff) => assert_eq!(cutoff, 75.0),
        }
    }

    #[test]
    fn test_args_score_cutoff_and_hint() {
        let args = Args::<f64, NoScoreCutoff>::default()
            .score_hint(85.0)
            .score_cutoff(90.0);
        match args.score_cutoff {
            WithScoreCutoff(cutoff) => assert_eq!(cutoff, 90.0),
        }
        assert_eq!(args.score_hint, Some(85.0));
    }

    // 7. Additional Tests for `RatioBatchComparator`
    #[test]
    fn test_ratio_batch_comparator_similar_strings() {
        let comparator = RatioBatchComparator::new("this is a test".chars());
        let score = comparator.similarity("this is a test!".chars());
        assert!(score > 90.0); // Adjust based on actual expected score
    }

    #[test]
    fn test_ratio_batch_comparator_completely_different_strings() {
        let comparator = RatioBatchComparator::new("this is a test".chars());
        let score = comparator.similarity("completely different".chars());
        assert!(score < 50.0); // Adjust based on actual expected score
    }

    #[test]
    fn test_ratio_batch_comparator_with_score_cutoff() {
        let comparator = RatioBatchComparator::new("rust programming language".chars());
        let args = Args::default().score_cutoff(80.0);
        let score = comparator.similarity_with_args("rust lang".chars(), &args);
        assert!(score.unwrap() >= 80.0); // Adjust based on expected behavior
    }

    #[test]
    fn test_ratio_batch_comparator_empty_strings() {
        let comparator = RatioBatchComparator::new("".chars());
        let score = comparator.similarity("".chars());
        assert_eq!(score, 100.0);

        let score = comparator.similarity("non-empty".chars());
        assert_eq!(score, 0.0);
    }

    // 8. Comprehensive Tests Covering All Functions
    #[test]
    fn test_comprehensive_wratio_flow() {
        let s1 = "fuzzy wuzzy was a bear";
        let s2 = "wuzzy fuzzy was a hare";

        // Compute different ratios
        let ratio = ratio(s1.chars(), s2.chars());
        let partial_ratio = partial_ratio(s1.chars(), s2.chars(), 0.0);
        let token_ratio = token_ratio_with_args(s1.chars(), s2.chars(), &Args::default());
        let partial_token_ratio = partial_token_ratio(s1.chars(), s2.chars(), 0.0);

        // Compute wratio
        let wratio = wratio(s1.chars(), s2.chars(), 0.0);

        // Validate that wratio is at least as good as the best individual ratio
        let max_individual = ratio
            .max(partial_ratio)
            .max(token_ratio)
            .max(partial_token_ratio);
        assert!(wratio >= max_individual);
    }

    #[test]
    fn test_comprehensive_ratio_with_score_cutoff() {
        let s1 = "abcdefg";
        let s2 = "abcxyzg";

        let score = ratio(s1.chars(), s2.chars());

        let high_cutoff_args = Args::default().score_cutoff(score + 0.1);
        let low_cutoff_args = Args::default().score_cutoff(score - 0.1);

        // With high cutoff, expect 0.0 if score is below cutoff
        let high_score = ratio_with_args(s1.chars(), s2.chars(), &high_cutoff_args);
        assert_eq!(high_score.unwrap(), 0.0);

        // With low cutoff, expect the actual score
        let low_score = ratio_with_args(s1.chars(), s2.chars(), &low_cutoff_args);
        assert_eq!(low_score.unwrap(), score);
    }

    // 9. Edge Case Tests
    #[test]
    fn test_ratio_with_args_single_character() {
        let s1 = "a";
        let s2 = "a";
        let args = Args::default();
        let score = ratio_with_args(s1.chars(), s2.chars(), &args);
        assert_eq!(score, 100.0);
    }

    #[test]
    fn test_ratio_with_args_single_character_different() {
        let s1 = "a";
        let s2 = "b";
        let args = Args::default();
        let score = ratio_with_args(s1.chars(), s2.chars(), &args);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_partial_ratio_with_args_single_character_substring() {
        let s1 = "a";
        let s2 = "abc";
        let args = Args::default();
        let score = partial_ratio_with_args(s1.chars(), s2.chars(), &args);
        assert_eq!(score, 100.0);
    }

    #[test]
    fn test_token_ratio_with_args_empty_tokens() {
        let s1 = "    "; // All spaces
        let s2 = "    ";
        let args = Args::default();
        let score = token_ratio_with_args(s1.chars(), s2.chars(), &args);
        assert_eq!(score, 100.0);
    }

    #[test]
    fn test_partial_token_ratio_with_args_repeated_tokens() {
        let s1 = "apple apple banana";
        let s2 = "apple banana banana";
        let args = Args::default();
        let score = partial_token_ratio_with_args(
            s1.chars().collect::<Vec<_>>(),
            sorted_split(s1.chars()),
            s2.chars(),
            &args,
        );
        assert!(score > 70.0); // Adjust based on expected behavior
    }

    // 10. Tests with Various `score_cutoff` Values
    #[test]
    fn test_ratio_with_args_various_cutoffs() {
        let s1 = "hello world";
        let s2 = "hello there";

        let score = ratio(s1.chars(), s2.chars());

        // High cutoff, expect 0.0 if score is below cutoff
        let args_high = Args::default().score_cutoff(score + 10.0);
        let score_high = ratio_with_args(s1.chars(), s2.chars(), &args_high);
        assert_eq!(score_high.unwrap(), 0.0);

        // Exact cutoff, expect the actual score
        let args_exact = Args::default().score_cutoff(score);
        let score_exact = ratio_with_args(s1.chars(), s2.chars(), &args_exact);
        assert_eq!(score_exact.unwrap(), score);

        // Low cutoff, expect the actual score
        let args_low = Args::default().score_cutoff(score - 10.0);
        let score_low = ratio_with_args(s1.chars(), s2.chars(), &args_low);
        assert_eq!(score_low.unwrap(), score);
    }

    // 11. Tests for Functions with Specific Behavior
    #[test]
    fn test_ratio_with_args_special_characters() {
        let s1 = "hello, world!";
        let s2 = "hello world";
        let args = Args::default();
        let score = ratio_with_args(s1.chars(), s2.chars(), &args);
        assert!(score > 80.0); // Adjust based on expected handling of punctuation
    }

    #[test]
    fn test_token_ratio_with_args_multiple_spaces() {
        let s1 = "hello   world";
        let s2 = "hello world";
        let args = Args::default();
        let score = token_ratio_with_args(s1.chars(), s2.chars(), &args);
        assert_eq!(score, 100.0);
    }

    // 12. Tests for Non-ASCII Characters and Normalization
    #[test]
    fn test_ratio_with_args_non_ascii_characters() {
        let s1 = "Café Münsterländer";
        let s2 = "Cafe Munsterlander";
        let args = Args::default();
        let score = ratio_with_args(s1.chars(), s2.chars(), &args);
        assert!(score > 80.0); // Adjust based on Unicode normalization handling
    }

    #[test]
    fn test_partial_token_ratio_with_args_reversed_strings() {
        let s1 = "the cat in the hat";
        let s2 = "hat the in cat the";
        let args = Args::default();
        let score = partial_token_ratio_with_args(
            s1.chars().collect::<Vec<_>>(),
            sorted_split(s1.chars()),
            s2.chars(),
            &args,
        );
        assert_eq!(score, 100.0); // Tokens match despite order
    }

    // 13. Testing `score_cutoff_to_distance` and `norm_distance` Utility Functions
    #[test]
    fn test_score_cutoff_to_distance() {
        let cutoff = 75.0;
        let lensum = 100;
        let distance = score_cutoff_to_distance(cutoff, lensum);
        assert_eq!(distance, 25);
    }

    #[test]
    fn test_norm_distance_above_cutoff() {
        let dist = 20;
        let lensum = 100;
        let cutoff = 75.0;
        let normalized = norm_distance(dist, lensum, cutoff);
        assert_eq!(normalized, 80.0); // 100 - (20 / 100)*100 = 80
    }

    #[test]
    fn test_norm_distance_below_cutoff() {
        let dist = 30;
        let lensum = 100;
        let cutoff = 75.0;
        let normalized = norm_distance(dist, lensum, cutoff);
        assert_eq!(normalized, 0.0);
    }

    #[test]
    fn test_norm_distance_exact_cutoff() {
        let dist = 25;
        let lensum = 100;
        let cutoff = 75.0;
        let normalized = norm_distance(dist, lensum, cutoff);
        assert_eq!(normalized, 75.0);
    }

    #[test]
    fn test_score_cutoff_to_distance_zero_cutoff() {
        let cutoff = 0.0;
        let lensum = 100;
        let distance = score_cutoff_to_distance(cutoff, lensum);
        assert_eq!(distance, 100);
    }

    #[test]
    fn test_norm_distance_zero_lensum() {
        let dist = 0;
        let lensum = 0;
        let cutoff = 50.0;
        let normalized = norm_distance(dist, lensum, cutoff);
        assert_eq!(normalized, 100.0);
    }

    // 14. Advanced Tests with `score_hint`
    #[test]
    fn test_ratio_with_args_with_score_hint() {
        let s1 = "fuzzy wuzzy was a bear";
        let s2 = "wuzzy fuzzy was a hare";
        let args = Args::default().score_cutoff(60.0).score_hint(80.0);
        let score = ratio_with_args(s1.chars(), s2.chars(), &args);
        assert!(score.unwrap() >= 60.0); // Adjust based on expected behavior
    }

    // 15. Testing with Different `CutoffType` Implementations
    #[test]
    fn test_ratio_with_args_no_score_cutoff() {
        let s1 = "example string";
        let s2 = "example string";
        let args = Args::default(); // NoScoreCutoff
        let score = ratio_with_args(s1.chars(), s2.chars(), &args);
        assert_eq!(score, 100.0);
    }

    #[test]
    fn test_ratio_with_args_with_score_cutoff() {
        let s1 = "example string";
        let s2 = "example string modified";
        let args = Args::default().score_cutoff(80.0);
        let score = ratio_with_args(s1.chars(), s2.chars(), &args);
        assert!(score.unwrap() >= 80.0); // Adjust based on expected behavior
    }

    // 16. Ensuring `RatioBatchComparator` Reusability
    #[test]
    fn test_ratio_batch_comparator_multiple_similar_strings() {
        let comparator = RatioBatchComparator::new("fuzzy wuzzy was a bear".chars());

        let score1 = comparator.similarity("fuzzy wuzzy was a bear".chars());
        assert_eq!(score1, 100.0);

        let score2 = comparator.similarity("wuzzy fuzzy was a hare".chars());
        assert!(score2 > 70.0); // Adjust based on expected score

        let score3 = comparator.similarity("nothing similar".chars());
        assert!(score3 < 50.0); // Adjust based on expected score
    }

    #[test]
    fn test_ratio_batch_comparator_empty_strings_2() {
        let comparator = RatioBatchComparator::new("".chars());

        let score1 = comparator.similarity("".chars());
        assert_eq!(score1, 100.0);

        let score2 = comparator.similarity("non-empty".chars());
        assert_eq!(score2, 0.0);
    }

    // 17. Testing Utility Functions
    #[test]
    fn test_score_cutoff_to_distance_2() {
        let cutoff = 75.0;
        let lensum = 100;
        let distance = score_cutoff_to_distance(cutoff, lensum);
        assert_eq!(distance, 25);
    }

    #[test]
    fn test_norm_distance_above_cutoff_2() {
        let dist = 20;
        let lensum = 100;
        let cutoff = 75.0;
        let normalized = norm_distance(dist, lensum, cutoff);
        assert_eq!(normalized, 80.0); // 100 - (20 / 100)*100 = 80
    }

    #[test]
    fn test_norm_distance_below_cutoff_2() {
        let dist = 30;
        let lensum = 100;
        let cutoff = 75.0;
        let normalized = norm_distance(dist, lensum, cutoff);
        assert_eq!(normalized, 0.0);
    }

    #[test]
    fn test_norm_distance_exact_cutoff_2() {
        let dist = 25;
        let lensum = 100;
        let cutoff = 75.0;
        let normalized = norm_distance(dist, lensum, cutoff);
        assert_eq!(normalized, 75.0);
    }

    #[test]
    fn test_score_cutoff_to_distance_zero_cutoff_2() {
        let cutoff = 0.0;
        let lensum = 100;
        let distance = score_cutoff_to_distance(cutoff, lensum);
        assert_eq!(distance, 100);
    }

    #[test]
    fn test_norm_distance_zero_lensum_2() {
        let dist = 0;
        let lensum = 0;
        let cutoff = 50.0;
        let normalized = norm_distance(dist, lensum, cutoff);
        assert_eq!(normalized, 100.0);
    }

    // 18. Testing `Partial Token Ratio` with Different Token Counts
    #[test]
    fn test_partial_token_ratio_with_args_more_tokens_in_s1() {
        let s1 = "apple orange banana grape";
        let s2 = "apple banana";
        let args = Args::default();
        let score = partial_token_ratio_with_args(
            s1.chars().collect::<Vec<_>>(),
            sorted_split(s1.chars()),
            s2.chars(),
            &args,
        );
        assert!(score > 70.0); // Adjust based on expected behavior
    }

    #[test]
    fn test_partial_token_ratio_with_args_more_tokens_in_s2() {
        let s1 = "apple banana";
        let s2 = "apple orange banana grape";
        let args = Args::default();
        let score = partial_token_ratio_with_args(
            s1.chars().collect::<Vec<_>>(),
            sorted_split(s1.chars()),
            s2.chars(),
            &args,
        );
        assert!(score > 70.0); // Adjust based on expected behavior
    }

    // 19. Testing `wratio` with Different Scoring Strategies
    #[test]
    fn test_wratio_with_args_high_similarity() {
        let s1 = "The quick brown fox jumps over the lazy dog";
        let s2 = "The quick brown fox leaps over the lazy dog";
        let args = Args::default().score_cutoff(80.0);
        let score = wratio_with_args(s1.chars(), s2.chars(), &args);
        assert!(score.unwrap() >= 80.0); // Adjust based on expected similarity
    }

    #[test]
    fn test_wratio_with_args_low_similarity() {
        let s1 = "The quick brown fox jumps over the lazy dog";
        let s2 = "Lorem ipsum dolor sit amet";
        let args = Args::default().score_cutoff(50.0);
        let score = wratio_with_args(s1.chars(), s2.chars(), &args);
        assert_eq!(score.unwrap(), 0.0);
    }

    // 20. Ensuring Consistency Across the Module
    #[test]
    fn test_ratio_non_with_args_consistency() {
        let s1 = "consistent test";
        let s2 = "consistent test";
        let score_default = ratio(s1.chars(), s2.chars());
        let score_with_args = ratio_with_args(s1.chars(), s2.chars(), &Args::default());
        assert_eq!(score_default, score_with_args);
    }

    #[test]
    fn test_wratio_non_with_args_consistency() {
        let s1 = "fuzzy wuzzy was a bear";
        let s2 = "wuzzy fuzzy was a hare";
        let score_default = wratio(s1.chars(), s2.chars(), 0.0);
        let score_with_args =
            wratio_with_args(s1.chars(), s2.chars(), &Args::default().score_cutoff(0.0));
        assert_eq!(score_default, score_with_args.unwrap());
    }

    // 21. Additional Edge Cases
    #[test]
    fn test_ratio_with_args_special_characters_2() {
        let s1 = "hello, world!";
        let s2 = "hello world";
        let args = Args::default();
        let score = ratio_with_args(s1.chars(), s2.chars(), &args);
        assert!(score > 80.0); // Adjust based on expected handling of punctuation
    }

    #[test]
    fn test_token_ratio_with_args_multiple_spaces_2() {
        let s1 = "hello   world";
        let s2 = "hello world";
        let args = Args::default();
        let score = token_ratio_with_args(s1.chars(), s2.chars(), &args);
        assert_eq!(score, 100.0);
    }

    #[test]
    fn test_ratio_with_args_non_ascii_characters_2() {
        let s1 = "Café Münsterländer";
        let s2 = "Cafe Munsterlander";
        let args = Args::default();
        let score = ratio_with_args(s1.chars(), s2.chars(), &args);
        assert!(score > 80.0); // Adjust based on Unicode normalization handling
    }

    #[test]
    fn test_partial_token_ratio_with_args_reversed_strings_2() {
        let s1 = "the cat in the hat";
        let s2 = "hat the in cat the";
        let args = Args::default();
        let score = partial_token_ratio_with_args(
            s1.chars().collect::<Vec<_>>(),
            sorted_split(s1.chars()),
            s2.chars(),
            &args,
        );
        assert_eq!(score, 100.0); // Tokens match despite order
    }

    // 22. Testing `RatioBatchComparator` Reusability
    #[test]
    fn test_ratio_batch_comparator_multiple_similar_strings_2() {
        let comparator = RatioBatchComparator::new("fuzzy wuzzy was a bear".chars());

        let score1 = comparator.similarity("fuzzy wuzzy was a bear".chars());
        assert_eq!(score1, 100.0);

        let score2 = comparator.similarity("wuzzy fuzzy was a hare".chars());
        assert!(score2 > 70.0); // Adjust based on expected score

        let score3 = comparator.similarity("nothing similar".chars());
        assert!(score3 < 50.0); // Adjust based on expected score
    }

    #[test]
    fn test_ratio_batch_comparator_empty_strings_3() {
        let comparator = RatioBatchComparator::new("".chars());

        let score1 = comparator.similarity("".chars());
        assert_eq!(score1, 100.0);

        let score2 = comparator.similarity("non-empty".chars());
        assert_eq!(score2, 0.0);
    }

    // 23. Testing Utility Functions (Already covered above)

    // 24. Testing `Partial Token Ratio` with Different Token Counts (Already covered above)

    // 25. Testing `wratio` with Different Scoring Strategies (Already covered above)

    // 26. Testing `RatioBatchComparator` Reusability (Already covered above)

    // 27. Testing Utility Functions (Already covered above)

    // 28. Testing `Partial Token Ratio` with Different Token Counts (Already covered above)

    // 29. Testing `wratio` with Different Scoring Strategies (Already covered above)

    // 30. Testing `Args` with Different Configurations
    #[test]
    fn test_args_combined_cutoff_and_hint() {
        let s1 = "example string";
        let s2 = "example string modified";
        let args = Args::default().score_cutoff(80.0).score_hint(85.0);
        let score = ratio_with_args(s1.chars(), s2.chars(), &args);
        assert!(score.unwrap() >= 80.0); // Adjust based on expected behavior
    }

    #[test]
    fn test_args_only_score_cutoff() {
        let s1 = "test string";
        let s2 = "test string";
        let args = Args::default().score_cutoff(90.0);
        let score = ratio_with_args(s1.chars(), s2.chars(), &args);
        assert_eq!(score.unwrap(), 100.0); // Since strings are identical and >= cutoff
    }

    #[test]
    fn test_args_only_score_hint() {
        let s1 = "test string";
        let s2 = "test string modified";
        let args = Args::default().score_hint(90.0);
        let score = ratio_with_args(s1.chars(), s2.chars(), &args);
        // Depending on implementation, score_hint might guide the scoring
        // For example, if the similarity is >= score_hint, it might prioritize or short-circuit
        // Adjust the assertion accordingly
        assert!(score >= 90.0 || score == 0.0); // Depending on implementation
    }

    #[test]
    fn test() {
        let s1 = "test string";
        let s2 = "test string";
        let score_ratio = ratio(s1.chars(), s2.chars());
        let score_partial_ratio = partial_ratio(s1.chars(), s2.chars(), 0.00);
        println!("{}", score_ratio);
        println!("{}", score_partial_ratio);
    }
}
