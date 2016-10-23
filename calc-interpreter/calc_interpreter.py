#! /usr/bin/python2.7
# -*- coding: utf-8 -*-

# @Author: v4if
# @Date:   2016-07-22 09:43:39
# @Last Modified by:   v4if
# @Last Modified time: 2016-10-23 14:32:55

"""
可以加括号，任意数量的加减乘除运算解析器
"""

# token types
INTEGER, PLUS, MINUS, MUL, DIV, LPAREN, RPAREN, EOF = 'INTEGER', 'PLUS', 'MINUS', 'MUL', 'DIV', 'LPAREN', 'RPAREN', 'EOF'


class Token(object):
    def __init__(self, type, value):
        super(Token, self).__init__()
        self.type = type
        self.value = value

    def __str__(self):
        return 'Token({type},{value})'.format(
            type=self.type,
            value=self.value
        )

    def __repr__(self):
        return self.__str__()


class Lexer(object):
    """scanner for input text"""

    def __init__(self, text):
        super(Lexer, self).__init__()
        self.text = text
        self.pos = 0
        self.current_char = text[self.pos]

    def error(self):
        raise Exception('invalid character')

    def advance(self):
        """ read input text """
        self.pos += 1
        if self.pos > len(self.text) - 1:
            self.current_char = None
        else:
            self.current_char = self.text[self.pos]

    def integer(self):
        result = ''
        while self.current_char is not None and self.current_char.isdigit():
            result += self.current_char
            self.advance()
        return int(result)

    def get_next_token(self):
        """ lexical analyzer """
        while self.current_char is not None:
            if self.current_char.isspace():
                self.advance()
                continue
            if self.current_char.isdigit():
                return Token(INTEGER, int(self.integer()))
            elif self.current_char == '+':
                self.advance()
                return Token(PLUS, '+')
            elif self.current_char == '-':
                self.advance()
                return Token(MINUS, '-')
            elif self.current_char == '*':
                self.advance()
                return Token(MUL, '*')
            elif self.current_char == '/':
                self.advance()
                return Token(DIV, '/')
            elif self.current_char == '(':
                self.advance()
                return Token(LPAREN, '(')
            elif self.current_char == ')':
                self.advance()
                return Token(RPAREN, ')')
            self.error()
        return Token(EOF, None)


class Interpreter(object):
    def __init__(self, lexer):
        super(Interpreter, self).__init__()
        self.lexer = lexer
        self.current_token = self.lexer.get_next_token()

    def error(self):
        raise Exception('Error parsing input')

    def eat(self, token_type):
        """ eat and get next token """
        if self.current_token.type == token_type:
            self.current_token = self.lexer.get_next_token()
        else:
            self.error()

    def factor(self):
        token = self.current_token
        if token.type == INTEGER:
        	self.eat(INTEGER)
        	return token.value
        elif token.type == LPAREN:
        	self.eat(LPAREN)
        	result = self.expr()
        	self.eat(RPAREN)
        	return result
        

    def term(self):
        result = self.factor()
        while self.current_token.type in (MUL, DIV):
            token = self.current_token
            if token.type == MUL:
                self.eat(MUL)
                result = result * self.factor()
            elif token.type == DIV:
                self.eat(DIV)
                result = result / self.factor()

        return result

    def expr(self):
        result = self.term()
        while self.current_token.type in (PLUS, MINUS):
            token = self.current_token
            if token.type == PLUS:
                self.eat(PLUS)
                result = result + self.term()
            elif token.type == MINUS:
                self.eat(MINUS)
                result = result - self.term()

        return result


def main():
    while True:
        try:
            text = raw_input("calc>")
        except EOFError:
            continue
        lexer = Lexer(text)
        interpreter = Interpreter(lexer)
        result = interpreter.expr()
        print(result)


if __name__ == '__main__':
    main()
