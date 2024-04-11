use std::{fmt::Display, iter::Peekable};

macro_rules! single_symbol_token {
    ($c:ident, $token:expr, $symbol:expr) => {
        if $c == $symbol {
            return Some($token);
        }
    };
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Token {
    Number(f64),
    Ident(String),
    NewLine,
    LeftParen,
    RightParen,
    LeftBracket,
    RightBracket,
    Comma,
    Semicolon,
    Equal,
    Plus,
    Minus,
    Asterisk,
    Slash,
    Circumflex,
}

impl Token {
    pub(super) fn priority(&self) -> Option<u8> {
        match self {
            Token::Number(_) => None,
            Token::Ident(_) => None,
            Token::NewLine => None,
            Token::LeftParen => None,
            Token::RightParen => None,
            Token::LeftBracket => None,
            Token::RightBracket => None,
            Token::Comma => None,
            Token::Semicolon => None,
            Token::Equal => None,
            Token::Plus => Some(1),
            Token::Minus => Some(2),
            Token::Asterisk => Some(3),
            Token::Slash => Some(4),
            Token::Circumflex => Some(5),
        }
    }
}

impl Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Number(num) => write!(f, "{num}"),
            Token::Ident(ident) => write!(f, "{ident}"),
            Token::NewLine => writeln!(f),
            Token::LeftParen => write!(f, "("),
            Token::RightParen => write!(f, ")"),
            Token::LeftBracket => write!(f, "["),
            Token::RightBracket => write!(f, "]"),
            Token::Comma => write!(f, ","),
            Token::Semicolon => write!(f, ";"),
            Token::Equal => write!(f, "="),
            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Asterisk => write!(f, "*"),
            Token::Slash => write!(f, "/"),
            Token::Circumflex => write!(f, "^"),
        }
    }
}

pub(super) struct Tokenizer<I: Iterator<Item = char>> {
    iter: Peekable<I>,
}

impl<I: Iterator<Item = char>> Tokenizer<I> {
    pub(super) fn new(iter: I) -> Self {
        Self {
            iter: iter.peekable(),
        }
    }

    fn parse_number(iter: &mut Peekable<I>, c: &char) -> Option<Token> {
        if c.is_numeric() || *c == '.' {
            let mut number = c.to_string();

            while let Some(digit) = iter.peek() {
                if digit.is_numeric()
                    || *digit == '.'
                    || *digit == '_'
                    || *digit == 'e'
                    || (number.ends_with('e') && *digit == '-')
                {
                    number.push(*digit);
                    iter.next();
                } else {
                    break;
                }
            }

            return Some(Token::Number(number.parse::<f64>().unwrap()));
        }
        None
    }
}

impl<I: Iterator<Item = char>> Iterator for Tokenizer<I> {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        let mut c = self.iter.next()?;

        if c == '-' {
            if let Some(c2) = self.iter.peek().copied() {
                if c2.is_numeric() || c2 == '.' {
                    match Self::parse_number(&mut self.iter, &c) {
                        Some(num) => return Some(num),
                        None => return Some(Token::Minus),
                    }
                }
            }
        }

        while c == ' ' || c == '\t' {
            c = self.iter.next()?;
        }

        if let Some(num) = Self::parse_number(&mut self.iter, &c) {
            return Some(num);
        }

        if c.is_alphabetic() {
            let mut identifier = c.to_string();

            while let Some(c) = self.iter.peek() {
                if c.is_alphanumeric() || *c == '_' {
                    identifier.push(*c);
                    self.iter.next();
                } else {
                    break;
                }
            }

            return Some(Token::Ident(identifier));
        }

        single_symbol_token!(c, Token::NewLine, '\n');
        single_symbol_token!(c, Token::LeftParen, '(');
        single_symbol_token!(c, Token::RightParen, ')');
        single_symbol_token!(c, Token::LeftBracket, '[');
        single_symbol_token!(c, Token::RightBracket, ']');
        single_symbol_token!(c, Token::Comma, ',');
        single_symbol_token!(c, Token::Semicolon, ';');
        single_symbol_token!(c, Token::Equal, '=');
        single_symbol_token!(c, Token::Plus, '+');
        single_symbol_token!(c, Token::Asterisk, '*');
        single_symbol_token!(c, Token::Slash, '/');
        single_symbol_token!(c, Token::Circumflex, '^');

        panic!("encountered invalid symbol \"{}\"", c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_tokenize() {
        let text = "f(x) = 3 * x + 1\nf(x) + y\n[1,2,3;4,5,6;7,8,9]\n-3e-5";
        let tokenizer = Tokenizer::new(text.chars());
        let tokens: Vec<Token> = tokenizer.collect();

        for token in tokens {
            print!("{}", token);
        }
        println!()
    }
}
