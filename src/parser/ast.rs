use std::{fmt::Display, iter::Peekable, ops::Deref};

use nalgebra::DMatrix;

use super::tokenizer::{Token, Tokenizer};

const DEBUG: bool = true;

macro_rules! debug_println {
    ($($arg:expr),*) => {
        if DEBUG {
            println!($($arg),*)
        }
    };
}

#[derive(Clone, Debug)]
pub(super) struct Ast {
    pub(super) sequence: Vec<Statement>,
}

impl Ast {
    pub(super) fn new<I: Iterator<Item = char>>(tokenizer: Tokenizer<I>) -> Option<Self> {
        let mut sequence = Vec::new();
        let mut tokenizer = tokenizer.peekable();

        while tokenizer.peek().is_some() {
            debug_println!("consuming while no new line");
            if let Some(ret) = Statement::consume_while::<_, consume_while::Line>(&mut tokenizer) {
                sequence.push(ret.0);
            }
        }

        Some(Self { sequence })
    }
}

impl Display for Ast {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for statement in &self.sequence {
            writeln!(f, "{}", statement)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Statement {
    Number(f64),
    Variable(String),
    Function(Function),
    Matrix(Matrix),
    Assignment(Box<BinaryOperation>),
    Add(Box<BinaryOperation>),
    Sub(Box<BinaryOperation>),
    Mul(Box<BinaryOperation>),
    Div(Box<BinaryOperation>),
    Pow(Box<BinaryOperation>),
}

impl Statement {
    fn consume_rhs<I: Iterator<Item = char>>(
        tokenizer: &mut Peekable<Tokenizer<I>>,
        priority: u8,
    ) -> Option<Self> {
        let mut next = match tokenizer.next()? {
            Token::Number(num) => Statement::Number(num),
            Token::Ident(ident) => Statement::identifier(ident, tokenizer),
            Token::LeftParen => Statement::consume_while::<_, consume_while::Group>(tokenizer)?.0,
            _ => panic!("invalid symbol after binary operator"),
        };

        if let Some(token) = tokenizer.peek() {
            if let Some(other_prio) = token.priority() {
                if other_prio > priority {
                    // Consume other
                    let token = tokenizer.next()?; // Binary operators are trivially copyable
                    let binop = BinaryOperation::new(tokenizer, next, other_prio)?;
                    next = match token {
                        Token::Plus => Statement::Add(Box::new(binop)),
                        Token::Minus => Statement::Sub(Box::new(binop)),
                        Token::Asterisk => Statement::Mul(Box::new(binop)),
                        Token::Slash => Statement::Div(Box::new(binop)),
                        Token::Circumflex => Statement::Pow(Box::new(binop)),
                        _ => unreachable!("encountered operator without priority"),
                    };
                }
            }
        }

        Some(next)
    }

    fn consume_while<I: Iterator<Item = char>, C: ConsumeWhile>(
        tokenizer: &mut Peekable<Tokenizer<I>>,
    ) -> Option<(Self, NextStep<C::Next>)> {
        let mut statements: Vec<Self> = Vec::new();
        let mut next_step = NextStep::Continue;

        while let Some(token) = tokenizer.next() {
            match C::check_token(&token) {
                NextStep::Continue => (),
                n => {
                    next_step = n;
                    break;
                }
            }

            match token {
                Token::Number(num) => statements.push(Statement::Number(num)),
                Token::Ident(ident) => statements.push(Statement::identifier(ident, tokenizer)),
                Token::Equal => {
                    debug_println!("found equal sign");
                    let lhs = statements
                        .pop()
                        .expect("encountered equal sign without left hand side");
                    debug_println!("left hand side: {:?}", lhs);
                    let rhs = Statement::consume_while::<_, consume_while::Line>(tokenizer)?.0;
                    debug_println!("right hand side: {:?}", rhs);
                    let binop = BinaryOperation { lhs, rhs };
                    statements.push(Statement::Assignment(Box::new(binop)));
                    break;
                }
                Token::Plus => {
                    debug_println!("found plus operator");
                    let lhs = statements
                        .pop()
                        .expect("encountered plus sign without left hand side");
                    let binop =
                        BinaryOperation::new(tokenizer, lhs, Token::Plus.priority().unwrap())?;
                    statements.push(Statement::Add(Box::new(binop)));
                }
                Token::Minus => {
                    debug_println!("found minus operator");
                    let lhs = statements
                        .pop()
                        .expect("encountered plus sign without left hand side");
                    let binop = BinaryOperation::new(tokenizer, lhs, token.priority().unwrap())?;
                    statements.push(Statement::Sub(Box::new(binop)));
                }
                Token::Asterisk => {
                    debug_println!("found multiplication operator");
                    let lhs = statements
                        .pop()
                        .expect("encountered plus sign without left hand side");
                    let binop = BinaryOperation::new(tokenizer, lhs, token.priority().unwrap())?;
                    statements.push(Statement::Mul(Box::new(binop)));
                }
                Token::Slash => {
                    debug_println!("found division operator");
                    let lhs = statements
                        .pop()
                        .expect("encountered plus sign without left hand side");
                    let binop = BinaryOperation::new(tokenizer, lhs, token.priority().unwrap())?;
                    statements.push(Statement::Div(Box::new(binop)));
                }
                Token::Circumflex => {
                    debug_println!("found power operator");
                    let lhs = statements
                        .pop()
                        .expect("encountered plus sign without left hand side");
                    let binop = BinaryOperation::new(tokenizer, lhs, token.priority().unwrap())?;
                    statements.push(Statement::Div(Box::new(binop)));
                }
                Token::LeftParen => (),
                Token::LeftBracket => statements.push(Statement::Matrix(Matrix::new(tokenizer))),
                Token::Comma
                | Token::Semicolon
                | Token::RightParen
                | Token::RightBracket
                | Token::NewLine => {
                    panic!("unexpected token: {}", token)
                }
            }
        }

        debug_println!("found statements {:?}", &statements);

        if statements.is_empty() {
            return None;
        } else if statements.len() != 1 {
            panic!("unexpected {} after {}", statements[1], statements[0]);
        }
        Some((statements.remove(0), next_step))
    }

    fn identifier<I: Iterator<Item = char>>(
        ident: String,
        tokenizer: &mut Peekable<Tokenizer<I>>,
    ) -> Self {
        debug_println!("parsing identifier {}", &ident);
        match tokenizer.peek() {
            Some(Token::LeftParen) => Statement::Function(Function::new(ident, tokenizer)),
            _ => Statement::Variable(ident),
        }
    }
}

impl Display for Statement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Statement::Number(num) => write!(f, "{num}")?,
            Statement::Variable(ident) => write!(f, "{ident}")?,
            Statement::Function(func) => {
                write!(f, "{}(", func.identifier)?;
                for (i, statement) in func.arguments.iter().enumerate() {
                    if i == 0 {
                        write!(f, "{}", statement)?;
                    } else {
                        write!(f, ", {}", statement)?;
                    }
                }
                write!(f, ")")?;
            }
            Statement::Matrix(mat) => write!(f, "{}", mat.deref())?,
            Statement::Assignment(binop) => write!(f, "{} = {}", binop.lhs, binop.rhs)?,
            Statement::Add(binop) => write!(f, "({} + {})", binop.lhs, binop.rhs)?,
            Statement::Sub(binop) => write!(f, "({} - {})", binop.lhs, binop.rhs)?,
            Statement::Mul(binop) => write!(f, "({} * {})", binop.lhs, binop.rhs)?,
            Statement::Div(binop) => write!(f, "({} / {})", binop.lhs, binop.rhs)?,
            Statement::Pow(binop) => write!(f, "({} ^ {})", binop.lhs, binop.rhs)?,
        }
        Ok(())
    }
}

#[derive(Copy, Clone, Debug)]
enum NextStep<T> {
    Continue,
    Terminate(T),
}

trait ConsumeWhile {
    type Next;

    fn check_token(token: &Token) -> NextStep<Self::Next>;
}

mod consume_while {
    use super::*;

    /// Consume a whole line
    #[derive(Copy, Clone, Debug)]
    pub(super) struct Line;

    impl ConsumeWhile for Line {
        type Next = ();

        fn check_token(token: &Token) -> NextStep<()> {
            if *token == Token::NewLine {
                NextStep::Terminate(())
            } else {
                NextStep::Continue
            }
        }
    }

    /// Consume arguments surrounded by parenthesis and separated by commas
    #[derive(Copy, Clone, Debug)]
    pub(super) struct Arguments;

    impl ConsumeWhile for Arguments {
        type Next = ArgumentsNext;

        fn check_token(token: &Token) -> NextStep<Self::Next> {
            if *token == Token::Comma {
                NextStep::Terminate(ArgumentsNext::Repeat)
            } else if *token == Token::RightParen {
                NextStep::Terminate(ArgumentsNext::Terminate)
            } else {
                NextStep::Continue
            }
        }
    }

    /// Consume group surrounded by parenthesis
    #[derive(Copy, Clone, Debug)]
    pub(super) struct Group;

    impl ConsumeWhile for Group {
        type Next = ();

        fn check_token(token: &Token) -> NextStep<Self::Next> {
            if *token == Token::RightParen {
                NextStep::Terminate(())
            } else {
                NextStep::Continue
            }
        }
    }

    /// Consume matrix
    #[derive(Copy, Clone, Debug)]
    pub(super) struct Matrix;

    impl ConsumeWhile for Matrix {
        type Next = MatrixNext;

        fn check_token(token: &Token) -> NextStep<Self::Next> {
            if *token == Token::RightBracket {
                NextStep::Terminate(MatrixNext::Terminate)
            } else if *token == Token::Semicolon {
                NextStep::Terminate(MatrixNext::RepeatAfterSemicolon)
            } else if *token == Token::Comma {
                NextStep::Terminate(MatrixNext::RepeatAfterComma)
            } else {
                NextStep::Continue
            }
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum ArgumentsNext {
    Repeat,
    Terminate,
}

#[derive(Copy, Clone, Debug)]
enum MatrixNext {
    RepeatAfterComma,
    RepeatAfterSemicolon,
    Terminate,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Function {
    pub(super) identifier: String,
    pub(super) arguments: Vec<Statement>,
}

impl Function {
    fn new<I: Iterator<Item = char>>(
        identifier: String,
        tokenizer: &mut Peekable<Tokenizer<I>>,
    ) -> Self {
        debug_println!("identifier was function. parsing arguments");
        // Remove leading left bracket
        tokenizer.next();

        let mut arguments = Vec::new();

        while tokenizer.peek().is_some() {
            if let Some(ret) = Statement::consume_while::<_, consume_while::Arguments>(tokenizer) {
                let (statement, next_step) = ret;
                arguments.push(statement);
                match next_step {
                    NextStep::Continue => unreachable!(),
                    NextStep::Terminate(next) => match next {
                        ArgumentsNext::Repeat => (),
                        ArgumentsNext::Terminate => break,
                    },
                }
            } else {
                break;
            }
        }

        debug_println!("arguments of function {} are {:?}", &identifier, &arguments);

        Self {
            identifier,
            arguments,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Matrix(pub(super) DMatrix<Statement>);

impl Matrix {
    fn new<I: Iterator<Item = char>>(tokenizer: &mut Peekable<Tokenizer<I>>) -> Self {
        debug_println!("identifier was matrix. parsing data");

        let mut data = Vec::new();
        let mut rows = 1;
        let mut cols = None;
        let mut curr_cols = 1;

        while tokenizer.peek().is_some() {
            if let Some(ret) = Statement::consume_while::<_, consume_while::Matrix>(tokenizer) {
                let (statement, next_step) = ret;
                data.push(statement);
                match next_step {
                    NextStep::Continue => unreachable!(),
                    NextStep::Terminate(next) => match next {
                        MatrixNext::RepeatAfterComma => {
                            curr_cols += 1;
                        }
                        MatrixNext::RepeatAfterSemicolon => {
                            rows += 1;
                            match cols {
                                Some(cols) => {
                                    assert_eq!(curr_cols, cols);
                                }
                                None => {
                                    cols = Some(curr_cols);
                                }
                            }
                            curr_cols = 1;
                        }
                        MatrixNext::Terminate => {
                            assert_eq!(cols.unwrap_or(1), curr_cols);
                            break;
                        }
                    },
                }
            } else {
                break;
            }
        }

        debug_println!(
            "found matrix with {} rows and {} columns",
            rows,
            cols.unwrap()
        );

        Self(DMatrix::from_vec(rows, cols.unwrap(), data))
    }
}

impl Deref for Matrix {
    type Target = DMatrix<Statement>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct BinaryOperation {
    pub(super) lhs: Statement,
    pub(super) rhs: Statement,
}

impl BinaryOperation {
    fn new<I: Iterator<Item = char>>(
        tokenizer: &mut Peekable<Tokenizer<I>>,
        lhs: Statement,
        priority: u8,
    ) -> Option<Self> {
        debug_println!("left hand side: {:?}", lhs);
        let rhs = if let Some(&Token::LeftParen) = tokenizer.peek() {
            tokenizer.next();
            Statement::consume_while::<_, consume_while::Group>(tokenizer)?.0
        } else {
            Statement::consume_rhs(tokenizer, priority)
                .expect("expected identifier or group after binary operator")
        };
        debug_println!("right hand side: {:?}", rhs);

        Some(Self { lhs, rhs })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_ast() {
        let text = "z = f(x + 5 / 2^2 - 7, 3, g(y))\nA = [1,2;3,4]";
        let tokenizer = Tokenizer::new(text.chars());
        let ast = Ast::new(tokenizer);
        assert!(ast.is_some());
        let ast = ast.unwrap();
        debug_println!("{}", ast);
    }
}
