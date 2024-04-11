mod ast;
mod parsing;
mod tokenizer;

use ast::Ast;
use tokenizer::Tokenizer;

pub fn parse(text: &str) {
    let tokenizer = Tokenizer::new(text.chars());
    let ast = Ast::new(tokenizer).unwrap();
    ast.traverse();
}
