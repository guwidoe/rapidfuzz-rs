use std::cmp::Ord;
use std::fmt::Debug;
use std::iter::Peekable;
use std::vec::Vec;

use crate::details::splitted_sentence::{is_space, IsSpace, SplittedSentence};
use crate::HashableChar;

/// Splits an input iterator into tokens based on whitespace, sorts them, and returns a `SplittedSentence`.
///
/// # Parameters
/// - `input`: An iterator over the input sequence.
///
/// # Returns
/// - A `SplittedSentence` containing sorted tokens.
///
/// # Notes
/// - Tokens are split based on whitespace characters determined by the `is_space` function.
/// - The function collects tokens into a vector of ranges or slices, sorts them, and constructs a `SplittedSentence`.
pub fn sorted_split<Iter, CharT>(input: Iter) -> SplittedSentence<CharT>
where
    Iter: IntoIterator<Item = CharT>,
    Iter::IntoIter: Clone + Iterator<Item = CharT>,
    CharT: IsSpace + HashableChar + Copy + Ord,
{
    let mut splitted: Vec<Vec<CharT>> = Vec::new();
    let mut iter = input.into_iter().peekable();

    while let Some(&ch) = iter.peek() {
        // Skip over any whitespace characters
        if is_space(ch) {
            iter.next();
            continue;
        }

        // Collect the token
        let mut token = Vec::new();
        while let Some(&ch) = iter.peek() {
            if is_space(ch) {
                break;
            }
            token.push(ch);
            iter.next();
        }

        if !token.is_empty() {
            splitted.push(token);
        }
    }

    // Sort the tokens
    splitted.sort();

    // Construct a SplittedSentence from the sorted tokens
    SplittedSentence::new(splitted)
}

#[derive(Default, Copy, Clone)]
pub struct NoScoreCutoff;
#[derive(Default, Copy, Clone)]
pub struct WithScoreCutoff<T>(pub T);

pub trait DistanceCutoff<T>
where
    T: Copy,
{
    type Output: Copy + Into<Option<T>> + PartialEq + Debug;

    fn cutoff(&self) -> Option<T>;
    fn score(&self, raw: T) -> Self::Output;
}

impl<T> DistanceCutoff<T> for NoScoreCutoff
where
    T: Copy + PartialEq + Debug,
{
    type Output = T;

    fn cutoff(&self) -> Option<T> {
        None
    }

    fn score(&self, raw: T) -> Self::Output {
        raw
    }
}

impl<T> DistanceCutoff<T> for WithScoreCutoff<T>
where
    T: Copy + PartialOrd + Debug,
{
    type Output = Option<T>;

    fn cutoff(&self) -> Option<T> {
        Some(self.0)
    }

    fn score(&self, raw: T) -> Self::Output {
        (raw <= self.0).then_some(raw)
    }
}

pub trait SimilarityCutoff<T>
where
    T: Copy,
{
    type Output: Copy + Into<Option<T>> + PartialEq + Debug;

    fn cutoff(&self) -> Option<T>;
    fn score(&self, raw: T) -> Self::Output;
}

impl<T> SimilarityCutoff<T> for NoScoreCutoff
where
    T: Copy + PartialEq + Debug,
{
    type Output = T;

    fn cutoff(&self) -> Option<T> {
        None
    }

    fn score(&self, raw: T) -> Self::Output {
        raw
    }
}

impl<T> SimilarityCutoff<T> for WithScoreCutoff<T>
where
    T: Copy + PartialOrd + Debug,
{
    type Output = Option<T>;

    fn cutoff(&self) -> Option<T> {
        Some(self.0)
    }

    fn score(&self, raw: T) -> Self::Output {
        (raw >= self.0).then_some(raw)
    }
}
