use nalgebra::DMatrix;

use crate::linalg::{
    eigen::{inverse_power_iteration, power_iteration, qr_algorithm},
    solve::Lu,
};

use super::ast::{Ast, Function, Matrix, Statement};

impl Ast {
    pub(super) fn traverse(self) {
        for statement in self.sequence {
            println!("{}", statement.eval());
        }
    }
}

impl Statement {
    pub(super) fn eval(self) -> Statement {
        match self {
            Statement::Function(func) => func.exec(),
            Statement::Number(_) => self,
            Statement::Matrix(matrix) => Statement::Matrix(matrix.eval()),
            _ => todo!(),
        }
    }

    pub(super) fn unwrap_matrix(self) -> Matrix {
        if let Statement::Matrix(matrix) = self {
            matrix
        } else {
            panic!("called unwrap_matrix on non-matrix statement");
        }
    }

    pub(super) fn unwrap_number(self) -> f64 {
        if let Statement::Number(number) = self {
            number
        } else {
            panic!("called unwrap_number on non-number statement");
        }
    }
}

impl Function {
    pub(super) fn exec(self) -> Statement {
        match BuiltInFunctions::from_identifier(&self.identifier) {
            Some(func) => func.eval(self.arguments),
            None => todo!(),
        }
    }
}

struct BuiltInFunction {
    argument_count: u8,
    argument_types: Vec<ArgumentType>,
}

enum ArgumentType {
    Number,
    Matrix,
}

impl ArgumentType {
    fn check_statement(&self, statement: &Statement) -> bool {
        match self {
            ArgumentType::Number => {
                matches!(statement, Statement::Number(_))
            }
            ArgumentType::Matrix => {
                matches!(statement, Statement::Matrix(_))
            }
        }
    }
}

enum BuiltInFunctions {
    MaxEigenvalue,
    Eigenvalue,
    AllEigenvalues,
    Solve,
}

impl BuiltInFunctions {
    fn from_identifier(ident: &str) -> Option<Self> {
        if ident == "max_eigenvalue" {
            Some(Self::MaxEigenvalue)
        } else if ident == "eigenvalue" {
            Some(Self::Eigenvalue)
        } else if ident == "all_eigenvalues" {
            Some(Self::AllEigenvalues)
        } else if ident == "solve" {
            Some(Self::Solve)
        } else {
            None
        }
    }

    fn eval(&self, mut args: Vec<Statement>) -> Statement {
        let n_args = args.len();
        let properties = self.properties();
        if properties.argument_count != n_args as u8 {
            panic!(
                "expected {} arguments, found {}",
                properties.argument_count, n_args
            );
        }

        for (arg, arg_type) in args.iter().zip(properties.argument_types.iter()) {
            assert!(arg_type.check_statement(arg));
        }

        match self {
            BuiltInFunctions::MaxEigenvalue => {
                let err = args.pop().unwrap().unwrap_number();
                let q0 = args
                    .pop()
                    .unwrap()
                    .unwrap_matrix()
                    .unwrap_numbers()
                    .fixed_columns(0)
                    .into();
                let a = args.pop().unwrap().unwrap_matrix().unwrap_numbers();

                let res = power_iteration(a, q0, err).unwrap().0;
                Statement::Number(res)
            }
            BuiltInFunctions::Eigenvalue => {
                let err = args.pop().unwrap().unwrap_number();
                let mu = args.pop().unwrap().unwrap_number();
                let q0 = args
                    .pop()
                    .unwrap()
                    .unwrap_matrix()
                    .unwrap_numbers()
                    .fixed_columns(0)
                    .into();
                let a = args.pop().unwrap().unwrap_matrix().unwrap_numbers();

                let res = inverse_power_iteration(a, q0, mu, err).unwrap().0;
                Statement::Number(res)
            }
            BuiltInFunctions::AllEigenvalues => {
                let n = args.pop().unwrap().unwrap_number();
                let a = args.pop().unwrap().unwrap_matrix().unwrap_numbers();

                let res = qr_algorithm(a, n as usize);
                let res = DMatrix::from_vec(
                    res.nrows(),
                    1,
                    res.into_iter().map(|x| Statement::Number(*x)).collect(),
                );
                Statement::Matrix(Matrix(res))
            }
            BuiltInFunctions::Solve => {
                let b = args
                    .pop()
                    .unwrap()
                    .unwrap_matrix()
                    .unwrap_numbers()
                    .fixed_columns(0)
                    .into();
                let a = args.pop().unwrap().unwrap_matrix().unwrap_numbers();

                let res = Lu::new(a).unwrap().solve(&b).unwrap();
                let res = DMatrix::from_vec(
                    res.nrows(),
                    1,
                    res.into_iter().map(|x| Statement::Number(*x)).collect(),
                );
                Statement::Matrix(Matrix(res))
            }
        }
    }

    fn properties(&self) -> BuiltInFunction {
        let argument_count;
        let argument_types;
        match self {
            BuiltInFunctions::MaxEigenvalue => {
                argument_count = 3;
                argument_types = vec![
                    ArgumentType::Matrix,
                    ArgumentType::Matrix,
                    ArgumentType::Number,
                ];
            }
            BuiltInFunctions::Eigenvalue => {
                argument_count = 4;
                argument_types = vec![
                    ArgumentType::Matrix,
                    ArgumentType::Matrix,
                    ArgumentType::Number,
                    ArgumentType::Number,
                ];
            }
            BuiltInFunctions::AllEigenvalues => {
                argument_count = 2;
                argument_types = vec![ArgumentType::Matrix, ArgumentType::Number];
            }
            BuiltInFunctions::Solve => {
                argument_count = 2;
                argument_types = vec![ArgumentType::Matrix, ArgumentType::Matrix];
            }
        }
        BuiltInFunction {
            argument_count,
            argument_types,
        }
    }
}

impl Matrix {
    fn eval(self) -> Self {
        let (rows, cols) = self.shape();
        let mut data: Vec<Statement> = self.0.data.into();
        data = data.into_iter().map(|statement| statement.eval()).collect();
        Self(DMatrix::from_vec(rows, cols, data))
    }

    fn unwrap_numbers(self) -> DMatrix<f64> {
        let (rows, cols) = self.shape();
        let data: Vec<_> = self.0.data.into();
        let data: Vec<f64> = data.into_iter().map(|s| s.unwrap_number()).collect();
        DMatrix::from_vec(rows, cols, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::tokenizer::Tokenizer;

    #[test]
    fn test_max_eigenvalue() {
        let text = "
            max_eigenvalue([1,2,3;4,5,6;7,8,9], [1;1;1], 1e-6)
            all_eigenvalues([1,2,3;4,5,6;7,8,9], 3)
            solve([1,2,3;4,5,6;7,8,7], [1;1;1])
        ";
        let tokenizer = Tokenizer::new(text.chars());
        let ast = Ast::new(tokenizer).unwrap();
        ast.traverse();
    }
}
