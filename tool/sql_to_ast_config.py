#!/usr/bin/env python3
"""
SQL -> AST -> metaphor-diagram JSON config (tool/config_*.json structure)

Parse a SQL statement into an abstract syntax tree, then follow the rules in `tool/sql_ast_json_rules.md`
and map it to a JSON config that `tool/sql_metaphor_diagram.py` can render directly:

    Nodes (rectangles) = relational nodes in the AST: base table / CTE / derived table / scalar subquery / result set
    Edges (arrows) = clauses and operations in the AST: FROM / JOIN ON / WHERE / GROUP BY / ORDER BY / LIMIT

Usage:
    python tool/sql_to_ast_config.py --sql-file tool/examples/artist_pipeline.sql --out tool/config_artist_pipeline.json
    python tool/sql_to_ast_config.py --sql "SELECT * FROM t WHERE a = 1" --out tool/config_demo.json
    python tool/sql_to_ast_config.py --sql-file x.sql --ast          # print AST only
    python tool/sql_to_ast_config.py --sql-file x.sql --out y.json --canvas 1400x760

This file uses only the Python standard library plus sqlglot (with a stdlib fallback parser).
"""

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ==================== Lexing ====================

KEYWORDS = {
    "SELECT", "FROM", "WHERE", "GROUP", "BY", "ORDER", "HAVING", "LIMIT", "OFFSET",
    "WITH", "RECURSIVE", "AS", "JOIN", "INNER", "LEFT", "RIGHT", "FULL", "OUTER",
    "CROSS", "NATURAL", "ON", "USING", "UNION", "INTERSECT", "EXCEPT", "ALL", "DISTINCT",
    "AND", "OR", "NOT", "IS", "NULL", "IN", "BETWEEN", "LIKE", "CASE", "WHEN", "THEN",
    "ELSE", "END", "CAST", "EXISTS", "ASC", "DESC", "NULLS", "FIRST", "LAST", "OVER",
    "PARTITION", "ROWS", "RANGE", "PRECEDING", "FOLLOWING", "CURRENT", "ROW",
    "UNBOUNDED", "INTERVAL", "TRUE", "FALSE", "UNKNOWN", "FILTER", "WINDOW",
}

# Words that must be treated as clause keywords (not aliases) after a SELECT-list expression
ALIAS_STOP_WORDS = KEYWORDS

JOIN_WORDS = {"JOIN", "INNER", "LEFT", "RIGHT", "FULL", "OUTER", "CROSS", "NATURAL"}

TWO_CHAR_OPS = ("<=", ">=", "<>", "!=", "||", "::", "=>", "->", ">>", "<<", "==")

SINGLE_CHAR = "(),.;*=<>+-/%:|[]"


class Token:
    __slots__ = ("kind", "text", "start", "end")

    def __init__(self, kind: str, text: str, start: int, end: int):
        self.kind = kind
        self.text = text
        self.start = start
        self.end = end

    def __repr__(self) -> str:  # pragma: no cover
        return "Token(%s,%r)" % (self.kind, self.text)


class ParseError(Exception):
    pass


def _strip_quotes(text: str) -> str:
    """Strip identifier quotes: `x` / "x" / [x]"""
    if len(text) >= 2 and text[0] in "`\"[":
        inner = text[1:-1]
        if text[0] == "`":
            return inner.replace("``", "`")
        if text[0] == '"':
            return inner.replace('""', '"')
        return inner
    return text


def tokenize(sql: str) -> List[Token]:
    tokens: List[Token] = []
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch.isspace():
            i += 1
            continue
        # line comment
        if ch == "-" and i + 1 < n and sql[i + 1] == "-":
            while i < n and sql[i] not in "\r\n":
                i += 1
            continue
        # block comment
        if ch == "/" and i + 1 < n and sql[i + 1] == "*":
            j = sql.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        # string / double-quoted identifier
        if ch in ("'", '"'):
            quote = ch
            start = i
            i += 1
            while i < n:
                if sql[i] == quote:
                    if i + 1 < n and sql[i + 1] == quote:
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            kind = "str" if quote == "'" else "qident"
            tokens.append(Token(kind, sql[start:i], start, i))
            continue
        # backtick identifier
        if ch == "`":
            start = i
            i += 1
            while i < n:
                if sql[i] == "`":
                    if i + 1 < n and sql[i + 1] == "`":
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            tokens.append(Token("qident", sql[start:i], start, i))
            continue
        # bracket identifier
        if ch == "[":
            j = sql.find("]", i + 1)
            j = n if j < 0 else j + 1
            tokens.append(Token("qident", sql[i:j], i, j))
            i = j
            continue
        # number
        if ch.isdigit() or (ch == "." and i + 1 < n and sql[i + 1].isdigit()):
            start = i
            while i < n and (sql[i].isdigit() or sql[i] == "."):
                i += 1
            if i < n and sql[i] in "eE":
                i += 1
                if i < n and sql[i] in "+-":
                    i += 1
                while i < n and sql[i].isdigit():
                    i += 1
            tokens.append(Token("num", sql[start:i], start, i))
            continue
        # identifier / keyword
        if ch.isalpha() or ch == "_" or ch == "$" or ord(ch) > 127:
            start = i
            while i < n:
                c = sql[i]
                if c.isalnum() or c in "_$" or ord(c) > 127:
                    i += 1
                    continue
                break
            word = sql[start:i]
            kind = "kw" if word.upper() in KEYWORDS else "ident"
            tokens.append(Token(kind, word, start, i))
            continue
        # two-character operator
        two = sql[i:i + 2]
        if two in TWO_CHAR_OPS:
            tokens.append(Token("op", two, i, i + 2))
            i += 2
            continue
        # single character
        if ch in SINGLE_CHAR:
            tokens.append(Token("op", ch, i, i + 1))
            i += 1
            continue
        # unknown character: skip to avoid failing the whole SQL parse
        i += 1
    return tokens


# ==================== Syntax: basic utilities ====================

class Parser:
    def __init__(self, sql: str):
        self.sql = sql
        self.tokens = tokenize(sql)
        self.i = 0

    # ---------- Cursor ----------
    def peek(self, k: int = 0) -> Optional[Token]:
        j = self.i + k
        return self.tokens[j] if j < len(self.tokens) else None

    def at_end(self) -> bool:
        return self.i >= len(self.tokens)

    def advance(self) -> Optional[Token]:
        tok = self.peek()
        self.i += 1
        return tok

    def raw(self, kind: str, text: str) -> bool:
        tok = self.peek()
        return bool(tok and tok.kind == kind and tok.text == text)

    def raw_advance(self, kind: str, text: str) -> bool:
        if self.raw(kind, text):
            self.i += 1
            return True
        return False

    def check_kw(self, word: str) -> bool:
        tok = self.peek()
        return bool(tok and tok.kind == "kw" and tok.text.upper() == word.upper())

    def accept_kw(self, word: str) -> bool:
        if self.check_kw(word):
            self.i += 1
            return True
        return False

    def expect_kw(self, word: str) -> Token:
        if not self.check_kw(word):
            tok = self.peek()
            got = tok.text if tok else "<EOF>"
            raise ParseError("expected keyword %r, got %r (token #%d)" % (word, got, self.i))
        return self.advance()

    def check_op(self, op: str) -> bool:
        tok = self.peek()
        return bool(tok and tok.kind == "op" and tok.text == op)

    def accept_op(self, op: str) -> bool:
        if self.check_op(op):
            self.i += 1
            return True
        return False

    def expect_op(self, op: str) -> Token:
        if not self.check_op(op):
            tok = self.peek()
            got = tok.text if tok else "<EOF>"
            raise ParseError("expected %r, got %r (token #%d)" % (op, got, self.i))
        return self.advance()

    def text_from(self, start_index: int) -> str:
        """Slice the original SQL by token offset so labels and expressions stay faithful to the source."""
        if start_index >= len(self.tokens):
            return ""
        start = self.tokens[start_index].start
        end_index = self.i - 1
        if end_index < start_index:
            end_index = start_index
        end = self.tokens[end_index].end
        return self.sql[start:end].strip()

    # ---------- Names ----------
    def parse_ident(self) -> str:
        tok = self.peek()
        if tok is None:
            raise ParseError("expected identifier, reached end of SQL")
        if tok.kind == "qident":
            self.i += 1
            return _strip_quotes(tok.text)
        if tok.kind in ("ident", "kw"):
            self.i += 1
            return tok.text
        raise ParseError("expected identifier, got %r (token #%d)" % (tok.text, self.i))

    def parse_dotted_name(self) -> str:
        parts = [self.parse_ident()]
        while self.check_op(".") and self.peek(1) is not None and self.peek(1).kind in ("ident", "kw", "qident"):
            self.i += 1
            parts.append(self.parse_ident())
        return ".".join(parts)

    def next_is_name(self) -> bool:
        tok = self.peek()
        return bool(tok and tok.kind in ("ident", "qident"))

    def parse_alias(self) -> Optional[str]:
        """Optional alias: AS xxx or a bare identifier (bare alias cannot be a keyword)."""
        if self.accept_kw("AS"):
            tok = self.peek()
            if tok and tok.kind in ("ident", "qident", "str"):
                self.i += 1
                return _strip_quotes(tok.text)
            return self.parse_ident()
        tok = self.peek()
        if tok and tok.kind == "ident":
            self.i += 1
            return tok.text
        if tok and tok.kind == "qident":
            self.i += 1
            return _strip_quotes(tok.text)
        return None

    # ---------- Expressions ----------
    # node types: column / literal / star / function / binary / unary / cast /
    #           case / between / in / is / exists / subquery / list / paren

    def parse_expr(self) -> Dict:
        start = self.i
        node = self.parse_or()
        node["text"] = self.text_from(start)
        return node

    def parse_or(self) -> Dict:
        start = self.i
        left = self.parse_and()
        while self.check_kw("OR"):
            self.advance()
            right = self.parse_and()
            left = {"type": "binary", "op": "OR", "left": left, "right": right,
                    "text": self.text_from(start)}
        return left

    def parse_and(self) -> Dict:
        start = self.i
        left = self.parse_not()
        while self.check_kw("AND"):
            self.advance()
            right = self.parse_not()
            left = {"type": "binary", "op": "AND", "left": left, "right": right,
                    "text": self.text_from(start)}
        return left

    def parse_not(self) -> Dict:
        start = self.i
        if self.check_kw("NOT"):
            self.advance()
            operand = self.parse_not()
            return {"type": "unary", "op": "NOT", "expr": operand, "text": self.text_from(start)}
        return self.parse_comparison()

    def parse_comparison(self) -> Dict:
        start = self.i
        left = self.parse_additive()
        while True:
            tok = self.peek()
            if tok is None:
                break
            # [NOT] BETWEEN
            if self.check_kw("BETWEEN") or (self.check_kw("NOT") and self.peek(1) is not None
                                            and self.peek(1).kind == "kw"
                                            and self.peek(1).text.upper() == "BETWEEN"):
                negated = False
                if self.check_kw("NOT"):
                    self.advance()
                    negated = True
                self.advance()  # BETWEEN
                low = self.parse_additive()
                self.accept_kw("AND")
                high = self.parse_additive()
                left = {"type": "between", "expr": left, "low": low, "high": high,
                        "not": negated, "text": self.text_from(start)}
                continue
            # [NOT] IN
            if self.check_kw("IN") or (self.check_kw("NOT") and self.peek(1) is not None
                                       and self.peek(1).kind == "kw"
                                       and self.peek(1).text.upper() == "IN"):
                negated = False
                if self.check_kw("NOT"):
                    self.advance()
                    negated = True
                self.advance()  # IN
                paren_start = self.i
                self.expect_op("(")
                items: List[Dict] = []
                if self.check_kw("SELECT") or self.check_kw("WITH"):
                    sub = self.parse_query()
                    items.append({"type": "subquery", "query": sub, "text": ""})
                elif not self.check_op(")"):
                    items.append(self.parse_expr())
                    while self.accept_op(","):
                        items.append(self.parse_expr())
                self.expect_op(")")
                # record the subquery original text (including outer parens) for whole-segment label replacement
                if items and items[0].get("type") == "subquery":
                    items[0]["text"] = self.text_from(paren_start)
                left = {"type": "in", "expr": left, "items": items, "not": negated,
                        "text": self.text_from(start)}
                continue
            # [NOT] LIKE
            if self.check_kw("LIKE") or (self.check_kw("NOT") and self.peek(1) is not None
                                         and self.peek(1).kind == "kw"
                                         and self.peek(1).text.upper() == "LIKE"):
                negated = False
                if self.check_kw("NOT"):
                    self.advance()
                    negated = True
                self.advance()  # LIKE
                right = self.parse_additive()
                left = {"type": "binary", "op": "NOT LIKE" if negated else "LIKE",
                        "left": left, "right": right, "text": self.text_from(start)}
                continue
            # IS [NOT] NULL / TRUE / FALSE
            if self.check_kw("IS"):
                self.advance()
                negated = self.accept_kw("NOT")
                tok2 = self.peek()
                target = tok2.text.upper() if tok2 else ""
                self.advance()
                left = {"type": "is", "expr": left, "not": negated, "target": target,
                        "text": self.text_from(start)}
                continue
            # standard comparison operators
            if tok.kind == "op" and tok.text in ("=", "==", "<>", "!=", "<", "<=", ">", ">="):
                op = tok.text
                self.advance()
                right = self.parse_additive()
                left = {"type": "binary", "op": op, "left": left, "right": right,
                        "text": self.text_from(start)}
                continue
            break
        return left

    def parse_additive(self) -> Dict:
        start = self.i
        left = self.parse_multiplicative()
        while self.peek() is not None and self.peek().kind == "op" and self.peek().text in ("+", "-", "||"):
            op = self.advance().text
            right = self.parse_multiplicative()
            left = {"type": "binary", "op": op, "left": left, "right": right,
                    "text": self.text_from(start)}
        return left

    def parse_multiplicative(self) -> Dict:
        start = self.i
        left = self.parse_unary()
        while self.peek() is not None and self.peek().kind == "op" and self.peek().text in ("*", "/", "%"):
            op = self.advance().text
            right = self.parse_unary()
            left = {"type": "binary", "op": op, "left": left, "right": right,
                    "text": self.text_from(start)}
        return left

    def parse_unary(self) -> Dict:
        start = self.i
        tok = self.peek()
        if tok is not None and tok.kind == "op" and tok.text in ("-", "+", "~"):
            op = self.advance().text
            operand = self.parse_unary()
            return {"type": "unary", "op": op, "expr": operand, "text": self.text_from(start)}
        return self.parse_primary()

    def parse_primary(self) -> Dict:
        start = self.i
        tok = self.peek()
        if tok is None:
            raise ParseError("unexpected end of expression (token #%d)" % self.i)

        # parentheses: subquery or plain grouping
        if tok.kind == "op" and tok.text == "(":
            if self._paren_starts_query():
                self.advance()
                sub = self.parse_query()
                self.expect_op(")")
                return {"type": "subquery", "query": sub, "text": self.text_from(start)}
            self.advance()
            inner = self.parse_expr()
            self.expect_op(")")
            return {"type": "paren", "expr": inner, "text": self.text_from(start)}

        # string / number
        if tok.kind == "str":
            self.advance()
            return {"type": "literal", "kind": "string", "text": tok.text}
        if tok.kind == "num":
            self.advance()
            return {"type": "literal", "kind": "number", "text": tok.text}

        # wildcard
        if tok.kind == "op" and tok.text == "*":
            self.advance()
            return {"type": "star", "qualifier": None, "text": "*"}

        # CASE WHEN
        if tok.kind == "kw" and tok.text.upper() == "CASE":
            return self.parse_case()

        # EXISTS (subquery)
        if tok.kind == "kw" and tok.text.upper() == "EXISTS":
            self.advance()
            self.expect_op("(")
            sub = self.parse_query()
            self.expect_op(")")
            return {"type": "exists", "query": sub, "text": self.text_from(start)}

        # CAST(expr AS type)
        if tok.kind == "kw" and tok.text.upper() == "CAST":
            self.advance()
            self.expect_op("(")
            inner = self.parse_expr()
            self.accept_kw("AS")
            type_parts = []
            depth = 0
            while not self.at_end():
                t2 = self.peek()
                if t2.kind == "op" and t2.text == ")" and depth == 0:
                    break
                if t2.kind == "op" and t2.text == "(":
                    depth += 1
                if t2.kind == "op" and t2.text == ")":
                    depth -= 1
                type_parts.append(self.advance().text)
            self.expect_op(")")
            return {"type": "cast", "expr": inner, "typename": " ".join(type_parts),
                    "text": self.text_from(start)}

        # identifier / function call / qualified column
        if tok.kind in ("ident", "qident", "kw"):
            parts = [self.parse_ident()]
            while self.check_op(".") and self.peek(1) is not None and self.peek(1).kind in ("ident", "kw", "qident"):
                self.i += 1
                parts.append(self.parse_ident())
            if self.check_op("("):
                return self.parse_function(parts)
            if len(parts) == 1:
                return {"type": "column", "qualifier": None, "name": parts[0],
                        "text": self.text_from(start)}
            return {"type": "column", "qualifier": ".".join(parts[:-1]), "name": parts[-1],
                    "text": self.text_from(start)}

        raise ParseError("unparseable expression %r (token #%d)" % (tok.text, self.i))

    def _paren_starts_query(self) -> bool:
        """Current token is '('; check whether it is followed by SELECT / WITH."""
        nxt = self.peek(1)
        if nxt is None or nxt.kind != "kw":
            return False
        return nxt.text.upper() in ("SELECT", "WITH")

    def parse_function(self, parts: List[str]) -> Dict:
        start = self.i - len(parts)
        start = self.i
        # start correction above: function name starts from the first token; text_from will override
        self.expect_op("(")
        args: List[Dict] = []
        distinct = self.accept_kw("DISTINCT")
        if not self.check_op(")"):
            args.append(self.parse_expr())
            while self.accept_op(","):
                args.append(self.parse_expr())
        self.expect_op(")")
        node = {"type": "function", "name": ".".join(parts), "args": args,
                "distinct": distinct, "window": None, "text": ""}
        if self.check_kw("OVER"):
            self.advance()
            node["window"] = self.parse_balanced_text()
        return node

    def parse_balanced_text(self) -> str:
        """Consume all tokens inside a pair of parentheses and return the original text (window definitions, etc.)."""
        self.expect_op("(")
        depth = 1
        parts = []
        while not self.at_end() and depth > 0:
            tok = self.advance()
            if tok.kind == "op" and tok.text == "(":
                depth += 1
            elif tok.kind == "op" and tok.text == ")":
                depth -= 1
                if depth == 0:
                    break
            parts.append(tok.text)
        return "(" + " ".join(parts) + ")"

    def parse_case(self) -> Dict:
        start = self.i
        self.advance()  # CASE
        base = None
        if not self.check_kw("WHEN"):
            base = self.parse_expr()
        branches = []
        while self.check_kw("WHEN"):
            self.advance()
            cond = self.parse_expr()
            self.accept_kw("THEN")
            result = self.parse_expr()
            branches.append({"when": cond, "then": result})
        else_expr = None
        if self.accept_kw("ELSE"):
            else_expr = self.parse_expr()
        self.accept_kw("END")
        return {"type": "case", "base": base, "branches": branches,
                "else": else_expr, "text": self.text_from(start)}

    # ---------- Statements / queries ----------
    def parse_statement(self) -> Dict:
        query = self.parse_query()
        while self.accept_op(";"):
            pass
        return query

    def parse_query(self) -> Dict:
        ctes: List[Dict] = []
        if self.check_kw("WITH"):
            self.advance()
            self.accept_kw("RECURSIVE")
            while True:
                name = self.parse_ident()
                columns: List[str] = []
                if self.check_op("(") and self._looks_like_column_list():
                    self.advance()
                    columns.append(self.parse_ident())
                    while self.accept_op(","):
                        columns.append(self.parse_ident())
                    self.expect_op(")")
                self.expect_kw("AS")
                self.expect_op("(")
                sub = self.parse_query()
                self.expect_op(")")
                ctes.append({"name": name, "columns": columns, "query": sub})
                if not self.accept_op(","):
                    break
        body = self.parse_select_core()
        while self.check_kw("UNION") or self.check_kw("INTERSECT") or self.check_kw("EXCEPT"):
            op = self.advance().text.upper()
            self.accept_kw("ALL")
            right = self.parse_select_core()
            body = {"type": "setop", "op": op, "left": body, "right": right}
        return {"ctes": ctes, "body": body}

    def _looks_like_column_list(self) -> bool:
        """Column list of WITH x (a, b) AS ...; distinguish from WITH x AS (SELECT ...)."""
        second = self.peek(1)
        third = self.peek(2)
        if second is None or second.kind not in ("ident", "qident", "kw"):
            return False
        return bool(third and third.kind == "op" and third.text in (",", ")"))

    def parse_select_core(self) -> Dict:
        if not self.accept_kw("SELECT"):
            tok = self.peek()
            got = tok.text if tok else "<EOF>"
            raise ParseError("expected SELECT, got %r (token #%d)" % (got, self.i))
        distinct = False
        if self.accept_kw("DISTINCT"):
            distinct = True
        else:
            self.accept_kw("ALL")

        select: List[Dict] = []
        while True:
            select.append(self.parse_select_item())
            if not self.accept_op(","):
                break

        from_items: List[Dict] = []
        if self.accept_kw("FROM"):
            from_items = self.parse_from_clause()

        where = None
        if self.accept_kw("WHERE"):
            where = self.parse_expr()

        group_by: List[Dict] = []
        if self.check_kw("GROUP"):
            self.advance()
            self.accept_kw("BY")
            group_by.append(self.parse_expr())
            while self.accept_op(","):
                group_by.append(self.parse_expr())

        having = None
        if self.accept_kw("HAVING"):
            having = self.parse_expr()

        order_by: List[Dict] = []
        if self.check_kw("ORDER"):
            self.advance()
            self.accept_kw("BY")
            order_by = self.parse_order_items()

        limit = None
        if self.accept_kw("LIMIT"):
            start = self.i
            self.parse_expr()
            if self.accept_op(","):
                self.parse_expr()
            if self.accept_kw("OFFSET"):
                self.parse_expr()
            limit = self.text_from(start)

        if self.check_kw("WINDOW"):
            # WINDOW clause is not visualised; skip its definition
            self.advance()
            while not self.at_end() and not self.check_op(";") and not self.check_op(")"):
                self.advance()

        return {
            "type": "select",
            "distinct": distinct,
            "select": select,
            "from": from_items,
            "where": where,
            "group_by": group_by,
            "having": having,
            "order_by": order_by,
            "limit": limit,
        }

    def parse_select_item(self) -> Dict:
        start = self.i
        tok = self.peek()
        # wildcard: * or t.*
        if tok is not None and tok.kind == "op" and tok.text == "*":
            self.advance()
            return {"expr": {"type": "star", "qualifier": None, "text": "*"},
                    "alias": None, "text": "*"}
        if (tok is not None and tok.kind in ("ident", "qident")
                and self.peek(1) is not None and self.peek(1).kind == "op"
                and self.peek(1).text == "."
                and self.peek(2) is not None and self.peek(2).kind == "op"
                and self.peek(2).text == "*"):
            qualifier = self.parse_ident()
            self.i += 2  # . *
            return {"expr": {"type": "star", "qualifier": qualifier,
                             "text": qualifier + ".*"},
                    "alias": None, "text": qualifier + ".*"}
        expr = self.parse_expr()
        alias = self.parse_alias()
        return {"expr": expr, "alias": alias, "text": self.text_from(start)}

    def parse_order_items(self) -> List[Dict]:
        items: List[Dict] = []
        while True:
            start = self.i
            expr = self.parse_expr()
            desc = False
            if self.accept_kw("DESC"):
                desc = True
            else:
                self.accept_kw("ASC")
            if self.accept_kw("NULLS"):
                self.advance()
            items.append({"expr": expr, "desc": desc, "text": self.text_from(start)})
            if not self.accept_op(","):
                break
        return items

    def parse_join_type(self) -> Optional[str]:
        tok = self.peek()
        if tok is None or tok.kind != "kw" or tok.text.upper() not in JOIN_WORDS:
            return None
        words: List[str] = []
        while True:
            t2 = self.peek()
            if t2 is None or t2.kind != "kw" or t2.text.upper() not in JOIN_WORDS:
                break
            words.append(self.advance().text.upper())
        if not words or words[-1] != "JOIN":
            return None
        return " ".join(words)

    def parse_from_clause(self) -> List[Dict]:
        items: List[Dict] = [self.parse_from_item()]
        while True:
            if self.accept_op(","):
                item = self.parse_from_item()
                item["join_type"] = ","
                items.append(item)
                continue
            join_type = self.parse_join_type()
            if join_type is None:
                break
            item = self.parse_from_item()
            item["join_type"] = join_type
            if self.accept_kw("ON"):
                item["on"] = self.parse_expr()
            elif self.accept_kw("USING"):
                self.expect_op("(")
                cols = [self.parse_ident()]
                while self.accept_op(","):
                    cols.append(self.parse_ident())
                self.expect_op(")")
                item["using"] = cols
            items.append(item)
        return items

    def parse_from_item(self) -> Dict:
        if self.check_op("(") and self._paren_starts_query():
            self.advance()
            sub = self.parse_query()
            self.expect_op(")")
            alias = self.parse_alias()
            return {"kind": "subquery", "query": sub, "alias": alias, "name": alias}
        name = self.parse_dotted_name()
        alias = self.parse_alias()
        return {"kind": "table", "name": name, "alias": alias}

# ==================== Utilities: expression traversal ====================

def unwrap(expr: Dict) -> Dict:
    """Strip parenthesisation wrapping."""
    node = expr
    while isinstance(node, dict) and node.get("type") == "paren":
        node = node.get("expr") or {}
    return node if isinstance(node, dict) else {}


def collect_columns(node, out: Optional[List[Dict]] = None) -> List[Dict]:
    """Collect column references in an expression; do not descend into subqueries (handled separately)."""
    if out is None:
        out = []
    if isinstance(node, dict):
        t = node.get("type")
        if t in ("subquery", "exists"):
            return out
        if t == "column":
            out.append(node)
            return out
        for key, value in node.items():
            if key == "text":
                continue
            collect_columns(value, out)
    elif isinstance(node, (list, tuple)):
        for value in node:
            collect_columns(value, out)
    return out


def collect_subqueries(node, out: Optional[List[Dict]] = None) -> List[Dict]:
    """Collect (top-level) subquery nodes; do not recurse inside subqueries."""
    if out is None:
        out = []
    if isinstance(node, dict):
        if node.get("type") in ("subquery", "exists"):
            out.append(node)
            return out
        for key, value in node.items():
            if key == "text":
                continue
            collect_subqueries(value, out)
    elif isinstance(node, (list, tuple)):
        for value in node:
            collect_subqueries(value, out)
    return out


def contains_subquery(node) -> bool:
    return bool(collect_subqueries(node))


def is_simple_column(expr: Dict) -> bool:
    return unwrap(expr).get("type") == "column"


def entity_name(display: str) -> str:
    """Take the last segment of a table name (strip the dataset prefix)."""
    return display.split(".")[-1]


def oneline(text: object) -> str:
    """Single-line a label: collapse newlines / consecutive whitespace."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


# ==================== AST -> config ====================

LAYOUT_CFG = {
    "mode": "layered",
    "gap_x": 46,
    "gap_y": 26,
    "margin_x": 34,
}


class ConfigBuilder:
    """Map the AST to a node/edge config following `sql_ast_json_rules.md`."""

    def __init__(self, ast: Dict, canvas: Tuple[int, int] = (1200, 620)):
        self.ast = ast
        self.canvas = canvas
        self.nodes: "OrderedDict[str, Dict]" = OrderedDict()
        self.roles: Dict[str, str] = {}
        self.edges: List[Dict] = []
        self.lineage: Dict[Tuple[str, str], set] = {}
        self.outputs: Dict[str, List[Dict]] = {}
        self.scopes: Dict[str, Dict] = {}
        self.cte_ids: Dict[str, str] = {}
        self.cte_headers: Dict[str, str] = {}
        self.global_index: Dict[str, str] = {}
        self.base_ids: Dict[str, str] = {}
        self.base_aliases: Dict[str, str] = {}
        self.base_split: Dict[str, str] = {}
        self.sub_count = 0
        self.derived_count = 0
        self.join_bend: Dict[str, int] = {}

    # ---------- Entities ----------
    def create_entity(self, node_id: str, header: str, role: str,
                      header_fill: Optional[str] = None,
                      header_color: Optional[str] = None) -> Dict:
        node: Dict = {"kind": "table", "header": header, "rows": []}
        if header_fill:
            node["header_fill"] = header_fill
        if header_color:
            node["header_color"] = header_color
        self.nodes[node_id] = node
        self.roles[node_id] = role
        return node

    def base_entity(self, name: str, alias: Optional[str]) -> str:
        """Each base table creates exactly one entity: the alias is taken from its first occurrence in the SQL;
        the same table reappearing with a different alias (self-join) triggers a node split."""
        short = entity_name(name)
        key = short.lower()
        if key in self.base_ids:
            existing = self.base_ids[key]
            existing_alias = (self.base_aliases.get(key) or "")
            if not alias or alias.lower() == existing_alias.lower():
                if alias:
                    self.global_index[alias.lower()] = existing
                self.global_index[name.lower()] = existing
                return existing
            if not existing_alias:
                # no alias first, alias later: merge into the same entity; header gets the SQL alias
                self.base_aliases[key] = alias
                self.nodes[existing]["header"] = short + " " + alias
                self.global_index[alias.lower()] = existing
                self.global_index[name.lower()] = existing
                return existing
            split_key = key + "|" + alias.lower()
            if split_key in self.base_split:
                return self.base_split[split_key]
            node_id = short + "_" + alias.lower()
            while node_id in self.nodes:
                node_id += "_"
            self.create_entity(node_id, short + " " + alias, "table")
            self.base_split[split_key] = node_id
            self.global_index[alias.lower()] = node_id
            return node_id
        node_id = short
        while node_id in self.nodes:
            node_id += "_"
        header = short + (" " + alias if alias else "")
        self.create_entity(node_id, header, "table")
        self.base_ids[key] = node_id
        self.base_aliases[key] = alias or ""
        self.global_index[short.lower()] = node_id
        self.global_index[name.lower()] = node_id
        if alias:
            self.global_index[alias.lower()] = node_id
        return node_id

    def ensure_from_item(self, item: Dict) -> str:
        if item.get("_entity"):
            return item["_entity"]
        if item.get("kind") == "subquery":
            self.derived_count += 1
            did = "d%d" % self.derived_count
            self.create_entity(did, "(subquery)", "derived")
            self.process_query(item["query"], did)
            if item.get("alias"):
                self.global_index[item["alias"].lower()] = did
            item["_entity"] = did
            return did
        name = item.get("name") or ""
        alias = item.get("alias")
        if "." not in name and name.lower() in self.cte_ids:
            eid = self.cte_ids[name.lower()]
            self.global_index[name.lower()] = eid
            if alias:
                self.global_index[alias.lower()] = eid
            item["_entity"] = eid
            return eid
        eid = self.base_entity(name, alias)
        item["_entity"] = eid
        return eid

    def build_scope(self, body: Dict) -> Dict:
        scope: Dict = {"by_name": {}, "default": None, "sources": [], "items": body.get("from", [])}
        for idx, item in enumerate(body.get("from", [])):
            eid = self.ensure_from_item(item)
            qualifiers = []
            if item.get("alias"):
                qualifiers.append(item["alias"])
            if item.get("name"):
                qualifiers.append(item["name"])
                qualifiers.append(entity_name(item["name"]))
            for qualifier in qualifiers:
                if qualifier:
                    scope["by_name"][qualifier.lower()] = eid
            if idx == 0:
                scope["default"] = eid
            scope["sources"].append(eid)
        if len(scope["sources"]) == 1:
            scope["default"] = scope["sources"][0]
        return scope

    def resolve_column(self, col: Dict, scope: Dict) -> Optional[str]:
        qualifier = (col.get("qualifier") or "").lower()
        if qualifier:
            if qualifier in scope["by_name"]:
                return scope["by_name"][qualifier]
            if qualifier in self.global_index:
                return self.global_index[qualifier]
            last = qualifier.split(".")[-1]
            if last in self.global_index:
                return self.global_index[last]
            return None
        return scope.get("default")

    def register_refs(self, expr, scope: Dict) -> None:
        """Register columns referenced by an expression into base-table entities' rows."""
        self.register_all([expr], scope)

    def register_all(self, exprs, scope: Dict) -> None:
        """Two-phase registration: qualified columns first, then unqualified ones (avoids mis-routing in multi-source scopes)."""
        exprs = [e for e in exprs if e]
        if len(scope.get("sources") or []) > 1:
            for expr in exprs:
                self._register_cols(expr, scope, qualified_only=True)
            for expr in exprs:
                self._register_cols(expr, scope, qualified_only=False)
        else:
            for expr in exprs:
                self._register_cols(expr, scope, qualified_only=None)

    def _owner_by_rows(self, name: str, scope: Dict) -> Optional[str]:
        """In a multi-source scope, determine unqualified-column ownership by existing rows; do not register when there is no unique owner."""
        if not name or name == "*":
            return None
        owners = []
        for eid in scope.get("sources") or []:
            node = self.nodes.get(eid) or {}
            if any(str(row).lower() == name.lower() for row in node.get("rows", [])):
                owners.append(eid)
        if len(owners) == 1:
            return owners[0]
        if len(owners) > 1:
            default = scope.get("default")
            return default if default in owners else None
        return None

    def _register_cols(self, expr, scope: Dict, qualified_only: Optional[bool]) -> None:
        for col in collect_columns(expr):
            qualifier = bool(col.get("qualifier"))
            if qualified_only is not None and qualifier != qualified_only:
                continue
            name = col.get("name") or ""
            if not name or name == "*":
                continue
            if qualified_only is False:
                eid = self._owner_by_rows(name, scope)
            else:
                eid = self.resolve_column(col, scope)
            if eid is None or self.roles.get(eid) != "table":
                continue
            rows = self.nodes[eid]["rows"]
            if name not in rows:
                rows.append(name)

    def has_attr(self, eid: str, name: str) -> bool:
        node = self.nodes.get(eid) or {}
        return any(str(row).lower() == str(name).lower() for row in node.get("rows", []))

    def output_name(self, expr: Dict) -> str:
        node = unwrap(expr)
        if node.get("type") == "column":
            return node.get("name") or "col"
        if node.get("type") == "star":
            return "*"
        return expr.get("text") or "expr"

    # ---------- Wiring helpers ----------
    def add_edge(self, edge: Dict) -> Dict:
        if edge.get("label"):
            edge["label"] = oneline(edge["label"])
        key = (edge.get("from"), edge.get("to"), edge.get("from_attr"),
               edge.get("to_attr"), edge.get("label"))
        for existing in self.edges:
            if (existing.get("from"), existing.get("to"), existing.get("from_attr"),
                    existing.get("to_attr"), existing.get("label")) == key:
                return existing
        self.edges.append(edge)
        return edge

    def _is_upstream(self, node_id: str, target: str) -> bool:
        """Whether node_id is upstream of target (reachable via reverse BFS along existing from->to edges).
        Used to decide whether a joined table's data already flowed in via the first source, to avoid duplicate wires."""
        preds: Dict[str, List[str]] = {}
        for edge in self.edges:
            preds.setdefault(edge.get("to"), []).append(edge.get("from"))
        seen: set = set()
        stack = [target]
        while stack:
            cur = stack.pop()
            for p in preds.get(cur, []):
                if p == node_id:
                    return True
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return False

    def _pair(self, on_expr) -> Optional[Tuple[Dict, Dict]]:
        """Extract column references on both sides of ON a.x = b.y."""
        node = unwrap(on_expr) if on_expr else {}
        if node.get("type") != "binary" or node.get("op") not in ("=", "=="):
            return None
        left = unwrap(node.get("left") or {})
        right = unwrap(node.get("right") or {})
        if left.get("type") != "column" or right.get("type") != "column":
            return None
        return left, right

    def _lineage_set(self, expr, scope: Dict) -> set:
        result = set()
        for col in collect_columns(expr):
            eid = self.resolve_column(col, scope)
            if eid:
                result.add((eid, col.get("name") or ""))
        return result

    def display_name(self, node_id: str) -> str:
        node = self.nodes.get(node_id) or {}
        header = node.get("header", node_id)
        if self.roles.get(node_id) == "table":
            return entity_name(header.split(" ")[0])
        return entity_name(header)

    def node_alias(self, node_id: str) -> str:
        node = self.nodes.get(node_id) or {}
        header = node.get("header", node_id)
        if self.roles.get(node_id) == "table" and " " in header:
            return header.split(" ", 1)[1]
        return entity_name(header)

    def simplify_on(self, on_expr) -> str:
        """Simplify a JOIN condition: same-name keys write only the column name; for AND combinations take only the first equality key."""
        node = unwrap(on_expr) if on_expr else {}
        if node.get("type") == "binary" and node.get("op") == "AND":
            for part in (node.get("left"), node.get("right")):
                text = self.simplify_on(part)
                if text != oneline((part or {}).get("text", "")):
                    return text
            return oneline(node.get("text", ""))
        pair = self._pair(on_expr)
        if pair:
            left, right = pair
            if (left.get("name") or "").lower() == (right.get("name") or "").lower():
                return left.get("name") or ""
        return oneline((on_expr or {}).get("text", ""))

    def _and_parts(self, predicate) -> List[Dict]:
        """Split a top-level AND into conjuncts."""
        node = unwrap(predicate)
        if node.get("type") == "binary" and node.get("op") == "AND":
            return self._and_parts(node.get("left")) + self._and_parts(node.get("right"))
        return [node] if node else []

    def _walk_nodes(self, node):
        if isinstance(node, dict):
            yield node
            for value in node.values():
                for item in self._walk_nodes(value):
                    yield item
        elif isinstance(node, list):
            for value in node:
                for item in self._walk_nodes(value):
                    yield item

    def _conjunct_for(self, predicate, sub: Dict) -> Dict:
        """Locate the top-level AND conjunct that contains a given subquery."""
        for part in self._and_parts(predicate):
            if any(item is sub for item in self._walk_nodes(part)):
                return part
        return predicate or {}

    def _strip_qualifiers(self, text: str, scopes: List[Dict]) -> str:
        for scope in scopes:
            for qualifier in sorted((scope or {}).get("by_name", {}).keys(),
                                    key=len, reverse=True):
                if qualifier:
                    text = re.sub("(?i)\\b%s\\." % re.escape(qualifier), "", text)
        return text

    def short_where(self, predicate, sub: Dict, replacement: str,
                    scopes: List[Dict]) -> str:
        """Compress a predicate to one line: keep only the conjunct containing the subquery, replace the subquery text with a short name."""
        part = self._conjunct_for(predicate, sub)
        text = part.get("text") or ""
        sub_text = sub.get("text") or ""
        if sub.get("type") == "exists":
            replacement = "EXISTS (subquery)"
        if sub_text and sub_text in text:
            head = text.split(sub_text, 1)[0]
            if re.search("(?i)\\bIN\\s*$", head) and not replacement.startswith("("):
                replacement = "(" + replacement + ")"
            text = text.replace(sub_text, replacement)
        text = self._strip_qualifiers(text, scopes)
        return "WHERE " + oneline(text)

    def _is_passthrough(self, sub: Dict) -> bool:
        """Whether a scalar subquery merely 'takes one column from an entity' (fold into a single edge, no node)."""
        query = sub.get("query") or {}
        body = query.get("body") or {}
        if body.get("type") != "select":
            return False
        if body.get("group_by") or body.get("having") or body.get("limit") or body.get("order_by"):
            return False
        if len(body.get("from") or []) != 1:
            return False
        for sel in body.get("select", []):
            if not is_simple_column(sel.get("expr") or {}):
                return False
        return True

    def _sub_source(self, sub: Dict) -> Optional[str]:
        body = (sub.get("query") or {}).get("body") or {}
        items = body.get("from") or []
        if not items:
            return None
        return self.ensure_from_item(items[0])

    def _fold_correlated_probe(self, sub: Dict, outer_scope: Dict):
        """Foldability check for a correlated scalar subquery: return (body, local_scope, corr_entities)
        or None. Foldable when: a select body + exactly one base table in FROM + an outer correlated reference exists.
        build_scope is idempotent; creating the inner base-table entity during probing has no side effects."""
        query = sub.get("query") or {}
        body = query.get("body") or {}
        if body.get("type") == "setop":
            body = body.get("left") or {}
        if body.get("type") != "select":
            return None
        items = body.get("from") or []
        if len(items) != 1 or items[0].get("kind") == "subquery":
            return None
        local_scope = self.build_scope(body)
        inner_eid = local_scope.get("default")
        if inner_eid is None or self.roles.get(inner_eid) != "table":
            return None

        outputs = [sel.get("expr") for sel in body.get("select", [])]
        order_exprs = [o.get("expr") for o in body.get("order_by", [])]
        scan = [body.get("where")] + outputs + order_exprs

        corr_entities: List[str] = []
        for expr in scan:
            if not expr:
                continue
            for col in collect_columns(expr):
                qualifier = (col.get("qualifier") or "").lower()
                if not qualifier or qualifier in local_scope["by_name"]:
                    continue
                oeid = None
                if outer_scope:
                    oeid = outer_scope.get("by_name", {}).get(qualifier)
                if oeid is None:
                    oeid = self.global_index.get(qualifier)
                if oeid and oeid != inner_eid and oeid not in local_scope["sources"]:
                    if oeid not in corr_entities:
                        corr_entities.append(oeid)
        if not corr_entities:
            return None
        return body, local_scope, corr_entities

    def try_fold_correlated(self, sub: Dict, consumer_id: str, tag: str,
                            outer_scope: Dict, out: Optional[Dict] = None) -> bool:
        """Fold a correlated scalar subquery into its real base table, without creating a sub_N node:
        (SELECT Name FROM artists WHERE ArtistId = ArtistSales.ArtistId)
        -> artists node gets Name/ArtistId rows
          + correlation key-pair edge artists.ArtistId <-> ArtistSales.ArtistId (reuse existing attribute edge)
          + artists -> consumer labelled edge (tag + correlation condition)
        When out is passed (a SELECT-position output item), replace the subquery text in the consumer's rows with the inner column name.
        Return False when the fold conditions are not met; the caller falls back to make_sub_entity."""
        probe = self._fold_correlated_probe(sub, outer_scope)
        if probe is None:
            return False
        body, local_scope, corr_entities = probe
        inner_eid = local_scope["default"]
        outputs = [sel.get("expr") for sel in body.get("select", [])]
        order_exprs = [o.get("expr") for o in body.get("order_by", [])]

        # register inner-referenced columns into the real base table rows (artists gets Name / ArtistId)
        self.register_all(outputs + [body.get("where")] + order_exprs, local_scope)

        # correlation key-pair edge: inner.x <-> outer.y (arrowless arc; reuse an existing same-name attribute edge
        # and mark the correlation condition on that wire so the "correlated join" is visible in the diagram)
        where = body.get("where")
        corr_label = (self.short_where(where, sub, "(subquery)", [local_scope])
                      if where else None)
        pair_drawn = False
        corr_labeled = False
        for part in self._and_parts(where):
            node = unwrap(part)
            if node.get("type") != "binary" or node.get("op") != "=":
                continue
            left, right = node.get("left"), node.get("right")
            for inner_col, outer_col in ((left, right), (right, left)):
                ic, oc = unwrap(inner_col), unwrap(outer_col)
                if ic.get("type") != "column" or oc.get("type") != "column":
                    continue
                qualifier = (oc.get("qualifier") or "").lower()
                if not qualifier or qualifier in local_scope["by_name"]:
                    continue
                oeid = None
                if outer_scope:
                    oeid = outer_scope.get("by_name", {}).get(qualifier)
                if oeid is None:
                    oeid = self.global_index.get(qualifier)
                if not oeid or oeid in local_scope["sources"]:
                    continue
                iqual = (ic.get("qualifier") or "").lower()
                if iqual and iqual not in local_scope["by_name"]:
                    continue
                # manually look up existing wires by (from,to,from_attr,to_attr) (ignoring label,
                # to avoid generating parallel edges on the second fold after the first fold annotated the wire)
                hit = None
                for e in self.edges:
                    if (e.get("from") == inner_eid and e.get("to") == oeid
                            and e.get("from_attr") == ic.get("name")
                            and e.get("to_attr") == oc.get("name")):
                        hit = e
                        break
                if hit is None:
                    hit = self.add_edge({
                        "from": inner_eid, "to": oeid, "kind": "curve",
                        "no_rank": True, "arrow": False,
                        "from_side": "right", "to_side": "right",
                        "from_attr": ic.get("name"), "to_attr": oc.get("name"),
                        "bend": [30, 0],
                    })
                # annotate the reused/new pair wire with the correlation condition (correlated-join visualisation)
                if corr_label and not hit.get("label"):
                    hit["label"] = corr_label
                if hit.get("label"):
                    corr_labeled = True
                pair_drawn = True
        if not pair_drawn:
            for oeid in corr_entities:
                if where:
                    label = self.short_where(where, sub, "(subquery)", [local_scope])
                else:
                    label = "FROM " + self.display_name(oeid)
                self.add_edge({"from": oeid, "to": inner_eid, "kind": "line",
                               "label": label})
                corr_labeled = True

        # subquery result -> consumer (do not repeat the annotation when the correlation condition is already on the pair wire)
        label = tag
        if where and not corr_labeled and not tag.startswith("WHERE"):
            label = tag + "; " + corr_label
        edge = {"from": inner_eid, "to": consumer_id, "kind": "line", "label": label}
        simple = [unwrap(e) for e in outputs if unwrap(e).get("type") == "column"]
        if len(outputs) == 1 and len(simple) == 1:
            edge["from_attr"] = simple[0].get("name")
        if out is not None and len(simple) == 1:
            inner_name = simple[0].get("name") or out["name"]
            edge["to_attr"] = inner_name
            old_name = out["name"]
            if old_name != inner_name:
                out["name"] = inner_name
                node = self.nodes.get(consumer_id)
                if node:
                    node["rows"] = [inner_name if r == old_name else r
                                    for r in node.get("rows", [])]
            self.lineage[(consumer_id, inner_name)] = {(inner_eid, edge["from_attr"])}
        self.add_edge(edge)
        return True

    def make_sub_entity(self, sub: Dict, consumer_id: str, tag: str,
                        outer_scope: Dict) -> str:
        """Materialise a scalar subquery: sub_N node + correlation-reference edge + sub->consumer edge."""
        self.sub_count += 1
        sid = "sub_%d" % self.sub_count
        query = sub.get("query") or {}
        body = query.get("body") or {}
        if body.get("type") == "setop":
            body = body.get("left") or {}
        local_scope = self.build_scope(body)
        outputs = []
        for sel in body.get("select", []):
            expr = sel.get("expr") or {}
            outputs.append({"name": sel.get("alias") or self.output_name(expr), "expr": expr})

        header = "(SELECT " + ", ".join(o["name"] for o in outputs) + ")"
        rows = []
        for item in body.get("from", []):
            if item.get("kind") == "subquery":
                rows.append("FROM (subquery)")
            else:
                rows.append("FROM " + entity_name(item.get("name") or ""))
        node = self.create_entity(sid, header, "sub")
        node["rows"] = rows or ["FROM ..."]

        self.register_all([o["expr"] for o in outputs] + [body.get("where")]
                          + [o.get("expr") for o in body.get("order_by", [])], local_scope)

        corr: List[Tuple[str, str]] = []
        scan = [body.get("where")] + [o["expr"] for o in outputs] + [o.get("expr") for o in body.get("order_by", [])]
        for expr in scan:
            if not expr:
                continue
            for col in collect_columns(expr):
                qualifier = (col.get("qualifier") or "").lower()
                if not qualifier or qualifier in local_scope["by_name"]:
                    continue
                eid = None
                if outer_scope:
                    eid = outer_scope["by_name"].get(qualifier)
                if eid is None:
                    eid = self.global_index.get(qualifier)
                if eid and eid != sid and eid not in local_scope["sources"]:
                    pair = (eid, col.get("name") or "")
                    if pair not in corr:
                        corr.append(pair)

        if corr:
            where = body.get("where")
            for eid, _ in corr:
                if where:
                    label = self.short_where(where, sub, "(subquery)", [local_scope])
                else:
                    label = "FROM " + self.display_name(eid)
                self.add_edge({"from": eid, "to": sid, "kind": "line", "label": label})
            self.add_edge({
                "from": sid, "to": consumer_id, "kind": "curve", "no_rank": True,
                "from_side": "right", "to_side": "right", "bend": [30, 0],
                "label": tag,
            })
        else:
            default = (outer_scope or {}).get("default")
            if default:
                self.add_edge({"from": default, "to": sid, "kind": "line", "label": tag})
                self.add_edge({
                    "from": sid, "to": consumer_id, "kind": "curve", "no_rank": True,
                    "from_side": "right", "to_side": "right", "bend": [30, 0],
                })
        return sid

    # ---------- Query -> nodes/edges ----------
    def process_query(self, query: Dict, consumer_id: str) -> None:
        # 0. local CTEs of a nested WITH
        for idx, cte in enumerate(query.get("ctes", [])):
            cid = "%s_c%d" % (consumer_id, idx + 1)
            if cid not in self.nodes:
                self.create_entity(cid, cte["name"], "cte")
            self.cte_ids[cte["name"].lower()] = cid
            self.global_index[cte["name"].lower()] = cid
        for idx, cte in enumerate(query.get("ctes", [])):
            cid = "%s_c%d" % (consumer_id, idx + 1)
            self.process_query(cte["query"], cid)

        body = query.get("body") or {}
        if body.get("type") == "setop":
            body = body.get("left") or {}

        # 1. scope (also creates entities from FROM)
        scope = self.build_scope(body)
        self.scopes[consumer_id] = scope

        # 2. output columns -> consumer entity rows
        outputs: List[Dict] = []
        for sel in body.get("select", []):
            expr = sel.get("expr") or {}
            name = sel.get("alias") or self.output_name(expr)
            outputs.append({"name": name, "expr": expr,
                            "text": sel.get("text") or name, "alias": sel.get("alias")})
        self.outputs[consumer_id] = outputs
        node = self.nodes.get(consumer_id)
        if node is None:
            node = self.create_entity(consumer_id,
                                      self.cte_headers.get(consumer_id, consumer_id), "cte")
        node["rows"] = [o["name"] for o in outputs] or ["*"]

        # 3. register referenced base-table columns (ON -> SELECT -> WHERE -> GROUP -> HAVING -> ORDER)
        refs: List = [item.get("on") for item in body.get("from", [])]
        refs += [out["expr"] for out in outputs]
        refs.append(body.get("where"))
        refs += list(body.get("group_by", []))
        refs.append(body.get("having"))
        refs += [order.get("expr") for order in body.get("order_by", [])]
        self.register_all(refs, scope)

        # USING (key) join keys are also registered into the participating base tables
        for item in body.get("from", []):
            for col in item.get("using") or []:
                for src in scope["sources"]:
                    if self.roles.get(src) != "table":
                        continue
                    rows = self.nodes[src]["rows"]
                    if not any(str(row).lower() == col.lower() for row in rows):
                        rows.append(col)

        items = body.get("from", [])
        has_group = bool(body.get("group_by"))

        # 4. JOIN key chain (only for "pure base-table scan chain + all simple two-column equalities"; result-set layer uses labelled JOIN edges)
        chain_ok = len(items) >= 2 and consumer_id != "result"
        if chain_ok:
            for item in items:
                eid = item.get("_entity")
                if not eid or self.roles.get(eid) != "table":
                    chain_ok = False
                    break
            if chain_ok:
                for item in items[1:]:
                    if not self._pair(item.get("on")):
                        chain_ok = False
                        break
        if chain_ok:
            for item in items[1:]:
                left, right = self._pair(item.get("on"))
                current = item["_entity"]
                a_entity = self.resolve_column(left, scope)
                b_entity = self.resolve_column(right, scope)
                if a_entity == current and b_entity and b_entity != current:
                    previous, from_col, to_col = b_entity, right.get("name"), left.get("name")
                elif b_entity == current and a_entity and a_entity != current:
                    previous, from_col, to_col = a_entity, left.get("name"), right.get("name")
                else:
                    continue
                self.add_edge({
                    "from": previous, "to": current, "kind": "curve",
                    "no_rank": True, "arrow": False,
                    "from_side": "right", "to_side": "right",
                    "from_attr": from_col, "to_attr": to_col, "bend": [34, 0],
                })

        connected: set = set()

        # 5. passthrough scalar subqueries in WHERE (register first; wire after projection edges)
        where_folds: List[Tuple[Dict, str]] = []
        where_covered = False
        for sub in collect_subqueries(body.get("where")):
            source = self._sub_source(sub)
            if source is None:
                continue
            if self._is_passthrough(sub):
                where_folds.append((sub, source))
                where_covered = True
                # inner columns of a folded subquery are also registered into its data-source entity
                inner = (sub.get("query") or {}).get("body") or {}
                local = self.build_scope(inner)
                self.register_all([sel.get("expr") for sel in inner.get("select", [])]
                                  + [inner.get("where")], local)
            elif self._fold_correlated_probe(sub, scope) is not None:
                # correlated subqueries are folded into labelled edges in step 7; WHERE no longer enters the clause label
                where_covered = True

        # 6. clause edges (GROUP BY / HAVING / ORDER BY / LIMIT / WHERE)
        clause_parts: List[str] = []
        if has_group:
            computed = []
            for out in outputs:
                kind = unwrap(out["expr"]).get("type")
                if kind in ("column", "star", "literal"):
                    continue
                computed.append(out["text"])
            if computed:
                clause_parts.append(", ".join(computed))
            clause_parts.append("GROUP BY " + ", ".join(
                group.get("text", "") for group in body.get("group_by", [])))
        if body.get("having"):
            clause_parts.append("HAVING " + (body["having"].get("text") or ""))
        order_plain = [o for o in body.get("order_by", [])
                       if not contains_subquery(o.get("expr"))]
        if order_plain:
            clause_parts.append("ORDER BY " + ", ".join(o.get("text", "") for o in order_plain))
        if body.get("limit"):
            clause_parts.append("LIMIT " + body["limit"])
        if body.get("where") and not where_covered:
            clause_parts.append("WHERE " + (body["where"].get("text") or ""))

        if clause_parts and scope["sources"]:
            primary = scope["sources"][0]
            self.add_edge({"from": primary, "to": consumer_id, "kind": "line",
                           "label": "; ".join(clause_parts)})
            connected.add(primary)

        # 7. scalar subqueries in other positions (SELECT / ORDER BY / HAVING):
        #    single base table + correlated reference -> fold to real base table; otherwise materialise as a sub node
        for out in outputs:
            expr = unwrap(out["expr"])
            if expr.get("type") == "subquery":
                if not self.try_fold_correlated(expr, consumer_id, "SELECT ...", scope, out):
                    self.make_sub_entity(expr, consumer_id, "SELECT ...", scope)
        for order in body.get("order_by", []):
            for sub in collect_subqueries(order.get("expr")):
                tag = "ORDER BY ... " + ("DESC" if order.get("desc") else "ASC")
                if not self.try_fold_correlated(sub, consumer_id, tag, scope):
                    self.make_sub_entity(sub, consumer_id, tag, scope)
        if body.get("having"):
            for sub in collect_subqueries(body["having"]):
                if not self.try_fold_correlated(sub, consumer_id, "HAVING ...", scope):
                    self.make_sub_entity(sub, consumer_id, "HAVING ...", scope)
        for sub in collect_subqueries(body.get("where")):
            if not self._is_passthrough(sub):
                tag = self.short_where(body["where"], sub, "(subquery)", [scope])
                if not self.try_fold_correlated(sub, consumer_id, tag, scope):
                    self.make_sub_entity(sub, consumer_id, tag, scope)

        # 8. projection / computed-expression edges (carried by clause edges when GROUP BY is present)
        for out in outputs:
            expr = unwrap(out["expr"])
            kind = expr.get("type")
            if kind == "star":
                # SELECT *, expr: the csv wildcard is carried by the other output edges, avoiding duplicate attribute-less edges
                if len(outputs) > 1:
                    continue
                targets = []
                if expr.get("qualifier"):
                    eid = scope["by_name"].get(str(expr["qualifier"]).lower())
                    if eid:
                        targets = [eid]
                else:
                    targets = list(OrderedDict.fromkeys(scope["sources"]))
                for eid in targets:
                    self.add_edge({"from": eid, "to": consumer_id,
                                   "kind": "line", "attr_link": True})
                    connected.add(eid)
                continue
            if kind == "subquery" or has_group:
                continue
            if kind == "column":
                eid = self.resolve_column(expr, scope)
                if eid is None or not self.has_attr(eid, expr["name"]):
                    continue
                out_name = out["name"]
                self.add_edge({"from": eid, "to": consumer_id, "kind": "line",
                               "from_attr": expr["name"], "to_attr": out_name})
                self.lineage[(consumer_id, out_name)] = {(eid, expr["name"])}
                connected.add(eid)
                continue
            if kind == "literal":
                continue
            refs = collect_columns(out["expr"])
            entities: List[str] = []
            for col in refs:
                eid = self.resolve_column(col, scope)
                if eid and eid not in entities:
                    entities.append(eid)
            if not entities:
                continue
            lineage_set = self._lineage_set(out["expr"], scope)
            self.lineage[(consumer_id, out["name"])] = lineage_set
            if len(entities) > 1:
                # cross-entity computation: the expression is only written into result-set rows; connection is carried by FROM / JOIN edges
                continue
            eid = entities[0]
            self.add_edge({"from": eid, "to": consumer_id, "kind": "line",
                           "label": out["text"]})
            connected.add(eid)

        if has_group:
            for out in outputs:
                expr = unwrap(out["expr"])
                if expr.get("type") != "column":
                    continue
                eid = self.resolve_column(expr, scope)
                if eid:
                    self.lineage[(consumer_id, out["name"])] = {(eid, expr["name"])}

        # 9. multi-source FROM: first-source FROM label + subsequent JOIN labels
        if len(items) > 1:
            first = items[0].get("_entity")
            if first and first not in connected:
                alias = items[0].get("alias") or self.node_alias(first)
                self.add_edge({"from": first, "to": consumer_id, "kind": "line",
                               "label": "FROM " + alias})
                connected.add(first)
            if not chain_ok:
                prev = items[0].get("_entity")
                for item in items[1:]:
                    eid = item.get("_entity")
                    if not eid:
                        continue
                    # USING join keys are visualised with arrowless arcs (same shape as the JOIN key chain)
                    if item.get("using") and prev and prev != eid:
                        for col in item["using"]:
                            self.add_edge({
                                "from": prev, "to": eid, "kind": "curve",
                                "no_rank": True, "arrow": False,
                                "from_side": "right", "to_side": "right",
                                "from_attr": col, "to_attr": col, "bend": [34, 0],
                            })
                    alias = item.get("alias") or self.node_alias(eid)
                    label = "JOIN " + alias
                    if item.get("on"):
                        label += " ON " + self.simplify_on(item["on"])
                    elif item.get("using"):
                        label += " USING (" + ", ".join(item["using"]) + ")"
                    # when a joined table is already upstream of the first source (its data already flowed into consumer via the first source),
                    # do not draw a separate line (otherwise the bottom curve would look like it originates from the last row of the first source, creating redundancy),
                    # merge the JOIN annotation into the first-source -> consumer edge label
                    if first and eid != first and self._is_upstream(eid, first):
                        merged = False
                        for edge in self.edges:
                            if (edge.get("from") == first and edge.get("to") == consumer_id
                                    and edge.get("label")):
                                edge["label"] = oneline(label + "; " + edge["label"])
                                merged = True
                                break
                        if not merged:
                            first_alias = items[0].get("alias") or self.node_alias(first)
                            self.add_edge({"from": first, "to": consumer_id, "kind": "line",
                                           "label": "FROM " + first_alias + "; " + label})
                    else:
                        self.add_edge({"from": eid, "to": consumer_id, "label": label,
                                       "_auto_join": True})
                    prev = eid

        # 10. fold edge for WHERE passthrough subqueries (drawn after projection / JOIN, label last)
        for sub, source in where_folds:
            label = self.short_where(body["where"], sub,
                                     self.display_name(source), [scope])
            self.add_edge({"from": source, "to": consumer_id,
                           "kind": "line", "label": label})

    # ---------- Finalisation: layered rank, highlight tracing, output ----------
    def compute_ranks(self) -> Dict[str, int]:
        preds = {node_id: set() for node_id in self.nodes}
        for edge in self.edges:
            if edge.get("no_rank"):
                continue
            a, b = edge.get("from"), edge.get("to")
            if a in self.nodes and b in self.nodes:
                preds[b].add(a)
        rank = {node_id: 0 for node_id in self.nodes}
        for _ in range(len(self.nodes) + 1):
            changed = False
            for node_id in self.nodes:
                if preds[node_id]:
                    value = 1 + max(rank[p] for p in preds[node_id])
                    if value > rank[node_id]:
                        rank[node_id] = value
                        changed = True
            if not changed:
                break
        for edge in self.edges:
            if edge.pop("_auto_join", False):
                a, b = edge["from"], edge["to"]
                gap = rank.get(b, 0) - rank.get(a, 0)
                index = self.join_bend.get(a, 0)
                self.join_bend[a] = index + 1
                if gap >= 2:
                    edge["kind"] = "curve"
                    edge["from_side"] = "bottom"
                    edge["to_side"] = "bottom"
                    edge["bend"] = [0, 90 + 50 * index]
                else:
                    edge["kind"] = "line"
        return rank

    def apply_highlights(self) -> None:
        """Trace upward from result-set expressions and mark columns participating in the final computation in red."""
        seen: set = set()
        result_scope = self.scopes.get("result") or {"by_name": {}, "default": None}
        for out in self.outputs.get("result", []):
            seen |= self._lineage_set(out["expr"], result_scope)
        frontier = list(seen)
        while frontier:
            item = frontier.pop()
            for source in self.lineage.get(item, ()):
                if source not in seen:
                    seen.add(source)
                    frontier.append(source)
        for eid, col in seen:
            node = self.nodes.get(eid)
            if not node or not col:
                continue
            match = None
            for row in node.get("rows", []):
                if str(row).lower() == str(col).lower():
                    match = row
                    break
            if match is not None:
                hl = node.setdefault("highlight_rows", [])
                if match not in hl:
                    hl.append(match)
        if "result" in self.nodes:
            self.nodes["result"]["highlight"] = True

    def build(self) -> Dict:
        # register CTE ids / headers first; entity nodes are created only when their query is processed:
        # so the node order is "base tables -> CTE chain -> result set", matching the in-column order of the layered layout
        for idx, cte in enumerate(self.ast.get("ctes", [])):
            cid = "L%d" % (idx + 1)
            self.cte_ids[cte["name"].lower()] = cid
            self.cte_headers[cid] = cte["name"]
            self.global_index[cte["name"].lower()] = cid

        for idx, cte in enumerate(self.ast.get("ctes", [])):
            cid = "L%d" % (idx + 1)
            self.process_query(cte["query"], cid)
            if cte.get("columns"):
                self.nodes[cid]["rows"] = list(cte["columns"])

        self.create_entity("result", "SELECT", "result",
                           header_fill="#cccccc", header_color="#666666")
        self.process_query({"ctes": [], "body": self.ast.get("body") or {}}, "result")

        self.compute_ranks()
        self.apply_highlights()

        return {
            "layout": dict(LAYOUT_CFG),
            "baseline_y": 320,
            "canvas": {
                "width": int(self.canvas[0]),
                "height": int(self.canvas[1]),
                "background": "#ffffff",
            },
            "nodes": self.nodes,
            "edges": self.edges,
        }


# ==================== sqlglot adapter ====================
# Prefer sqlglot to parse SQL, then adapt the sqlglot AST to the custom AST
# structure produced by the Parser above (ConfigBuilder stays unchanged). If sqlglot is unavailable or fails,
# fall back to the pure-stdlib Parser.

_BIN_OP_MAP = {
    "Add": "+", "Sub": "-", "Mul": "*", "Div": "/", "Mod": "%",
    "EQ": "=", "Neq": "<>", "GT": ">", "LT": "<", "GTE": ">=", "LTE": "<=",
    "And": "AND", "Or": "OR",
    "StringConcat": "||",
    "BitwiseAnd": "&", "BitwiseOr": "|", "BitwiseXor": "^",
}


def _sql_text(node) -> str:
    """Return the node SQL text, or an empty string on failure."""
    try:
        return node.sql()
    except Exception:
        return ""


def _arg_type_keys(node) -> list:
    """Compatibility across sqlglot versions: arg_types may be a dict or a tuple."""
    at = getattr(node, "arg_types", None)
    if isinstance(at, dict):
        return list(at.keys())
    if isinstance(at, (tuple, list)):
        return list(at)
    return []


def parse_sql_via_sqlglot(sql: str) -> Dict:
    """Parse SQL with sqlglot and convert to the custom AST structure."""
    import sqlglot  # noqa: F401
    from sqlglot import exp
    tree = sqlglot.parse_one(sql)
    return _convert_statement(tree, exp)


def _convert_statement(node, exp) -> Dict:
    # strip the Subquery wrapper layer (IN/EXISTS/scalar subquery scenarios)
    while isinstance(node, exp.Subquery):
        node = node.this
    ctes = _extract_ctes(node, exp)
    body = _convert_query_body(node, exp)
    return {"ctes": ctes, "body": body}


def _extract_ctes(node, exp) -> List[Dict]:
    ctes: List[Dict] = []
    if not isinstance(node, exp.Expression):
        return ctes
    with_node = node.args.get("with_")
    if with_node is None:
        return ctes
    for cte in getattr(with_node, "expressions", []) or []:
        alias_obj = cte.args.get("alias")
        if isinstance(alias_obj, str):
            name = alias_obj
            columns: List[str] = []
        elif alias_obj is not None:
            name = alias_obj.name
            columns = [c.name for c in (alias_obj.columns or []) if c is not None]
        else:
            name = ""
            columns = []
        ctes.append({
            "name": name,
            "columns": columns,
            "query": _convert_statement(cte.this, exp),
        })
    return ctes


def _convert_query_body(node, exp) -> Dict:
    if isinstance(node, (exp.Union, exp.Intersect, exp.Except)):
        op = type(node).__name__.upper()
        return {
            "type": "setop",
            "op": op,
            "left": _convert_query_body(node.this, exp),
            "right": _convert_query_body(node.expression, exp),
        }
    return _convert_select(node, exp)


def _convert_select(node, exp) -> Dict:
    select_items = [_convert_select_item(e, exp) for e in (node.expressions or [])]

    from_items: List[Dict] = []
    fr = node.args.get("from_")
    if fr is not None and getattr(fr, "this", None) is not None:
        from_items.append(_convert_from_item(fr.this, exp))
    for j in (node.args.get("joins") or []):
        from_items.append(_convert_join(j, exp))

    where = None
    w = node.args.get("where")
    if w is not None and getattr(w, "this", None) is not None:
        where = _convert_expr(w.this, exp)

    group_by: List[Dict] = []
    g = node.args.get("group")
    if g is not None:
        group_by = [_convert_expr(x, exp) for x in (g.expressions or [])]

    having = None
    h = node.args.get("having")
    if h is not None and getattr(h, "this", None) is not None:
        having = _convert_expr(h.this, exp)

    order_by: List[Dict] = []
    o = node.args.get("order")
    if o is not None:
        for item in (o.expressions or []):
            order_by.append({
                "expr": _convert_expr(item.this, exp),
                "desc": bool(item.args.get("desc")),
                "text": _sql_text(item),
            })

    limit = None
    lim = node.args.get("limit")
    if lim is not None:
        txt = _sql_text(lim)
        if txt[:6].upper() == "LIMIT ":
            txt = txt[6:]
        limit = txt

    distinct = bool(node.args.get("distinct"))

    return {
        "type": "select",
        "distinct": distinct,
        "select": select_items,
        "from": from_items,
        "where": where,
        "group_by": group_by,
        "having": having,
        "order_by": order_by,
        "limit": limit,
    }


def _convert_select_item(e, exp) -> Dict:
    text = _sql_text(e)
    if isinstance(e, exp.Alias):
        inner = e.this
        alias = e.alias or None
    else:
        inner = e
        alias = getattr(e, "alias", None)
        if alias == "":
            alias = None

    if isinstance(inner, exp.Star):
        expr = {"type": "star", "qualifier": None, "text": "*"}
    elif (isinstance(inner, exp.Column)
          and isinstance(getattr(inner, "this", None), exp.Star)):
        qualifier = inner.table
        expr = {"type": "star", "qualifier": qualifier, "text": _sql_text(inner)}
    else:
        expr = _convert_expr(inner, exp)
    return {"expr": expr, "alias": alias, "text": text}


def _convert_from_item(node, exp) -> Dict:
    if isinstance(node, exp.Subquery):
        alias = (node.alias or None)
        return {
            "kind": "subquery",
            "query": _convert_statement(node.this, exp),
            "alias": alias,
            "name": alias,
        }
    if isinstance(node, exp.Table):
        parts = []
        if node.catalog:
            parts.append(node.catalog)
        if node.db:
            parts.append(node.db)
        parts.append(node.name)
        name = ".".join(parts)
        alias = node.alias or None
        return {"kind": "table", "name": name, "alias": alias}
    # fallback
    return {"kind": "table", "name": _sql_text(node), "alias": None}


def _convert_join(j, exp) -> Dict:
    item = _convert_from_item(j.this, exp)
    kind = j.args.get("kind")
    side = j.args.get("side")
    on = j.args.get("on")
    using = j.args.get("using")
    has_condition = on is not None or using is not None

    if kind or side:
        parts = []
        if side:
            parts.append(str(side))
        if kind:
            parts.append(str(kind))
        item["join_type"] = " ".join(parts + ["JOIN"])
    elif not has_condition:
        item["join_type"] = ","
    else:
        item["join_type"] = "JOIN"

    if on is not None:
        item["on"] = _convert_expr(on, exp)
    if using is not None:
        cols = []
        for c in (getattr(using, "expressions", None) or []):
            if c is not None:
                cols.append(getattr(c, "name", _sql_text(c)))
        item["using"] = cols
    return item


def _convert_function(node, exp) -> Dict:
    if isinstance(node, exp.Anonymous):
        name = node.name
    else:
        name = node.sql_name()

    distinct = bool(node.args.get("distinct"))
    args: List = []
    for key in _arg_type_keys(node):
        val = node.args.get(key)
        if val is None:
            continue
        if isinstance(val, exp.Distinct):
            distinct = True
            inner = val.this
            if isinstance(inner, exp.Expression):
                args.append(inner)
            elif isinstance(inner, list):
                args.extend(inner)
        elif isinstance(val, exp.Expression):
            args.append(val)
        elif isinstance(val, list):
            args.extend(v for v in val if isinstance(v, exp.Expression))

    window = None
    over = node.args.get("over")
    if over is not None:
        window = _sql_text(over)

    return {
        "type": "function",
        "name": name,
        "args": [_convert_expr(a, exp) for a in args],
        "distinct": distinct,
        "window": window,
        "text": _sql_text(node),
    }


def _is_target(node, exp) -> str:
    if node is None:
        return ""
    if isinstance(node, exp.Null):
        return "NULL"
    if isinstance(node, exp.Boolean):
        return "TRUE" if node.this else "FALSE"
    return type(node).__name__.upper()


def _convert_expr(node, exp) -> Dict:
    if node is None:
        return None
    if isinstance(node, exp.Alias):
        return _convert_expr(node.this, exp)
    if isinstance(node, exp.Column):
        return {
            "type": "column",
            "qualifier": node.table or None,
            "name": node.name,
            "text": _sql_text(node),
        }
    if isinstance(node, exp.Literal):
        kind = "string" if node.is_string else "number"
        return {"type": "literal", "kind": kind, "text": _sql_text(node)}
    if isinstance(node, exp.Star):
        return {"type": "star", "qualifier": None, "text": "*"}
    if isinstance(node, exp.Null):
        return {"type": "literal", "kind": "null", "text": "NULL"}
    if isinstance(node, exp.Boolean):
        val = "TRUE" if node.this else "FALSE"
        return {"type": "literal", "kind": "bool", "text": val}
    if isinstance(node, exp.Cast):
        return {
            "type": "cast",
            "expr": _convert_expr(node.this, exp),
            "typename": _sql_text(node.to),
            "text": _sql_text(node),
        }
    if isinstance(node, exp.Paren):
        return {
            "type": "paren",
            "expr": _convert_expr(node.this, exp),
            "text": _sql_text(node),
        }
    if isinstance(node, exp.Not):
        inner = node.this
        if isinstance(inner, (exp.In, exp.Between, exp.Is)):
            converted = _convert_expr(inner, exp)
            converted["not"] = True
            return converted
        return {
            "type": "unary",
            "op": "NOT",
            "expr": _convert_expr(inner, exp),
            "text": _sql_text(node),
        }
    if isinstance(node, exp.In):
        items: List[Dict] = []
        q = node.args.get("query")
        if q is not None:
            items.append({
                "type": "subquery",
                "query": _convert_statement(q, exp),
                "text": _sql_text(q),
            })
        else:
            for x in (node.expressions or []):
                items.append(_convert_expr(x, exp))
        negated = bool(node.args.get("negated"))
        return {
            "type": "in",
            "expr": _convert_expr(node.this, exp),
            "items": items,
            "not": negated,
            "text": _sql_text(node),
        }
    if isinstance(node, exp.Between):
        negated = bool(node.args.get("negated"))
        return {
            "type": "between",
            "expr": _convert_expr(node.this, exp),
            "low": _convert_expr(node.args.get("low"), exp),
            "high": _convert_expr(node.args.get("high"), exp),
            "not": negated,
            "text": _sql_text(node),
        }
    if isinstance(node, exp.Is):
        target = _is_target(node.args.get("expression"), exp)
        negated = bool(node.args.get("negated"))
        return {
            "type": "is",
            "expr": _convert_expr(node.this, exp),
            "not": negated,
            "target": target,
            "text": _sql_text(node),
        }
    if isinstance(node, exp.Exists):
        return {
            "type": "exists",
            "query": _convert_statement(node.this, exp),
            "text": _sql_text(node),
        }
    if isinstance(node, exp.Subquery):
        return {
            "type": "subquery",
            "query": _convert_statement(node.this, exp),
            "text": _sql_text(node),
        }
    if isinstance(node, exp.Case):
        base = _convert_expr(node.this, exp) if node.this is not None else None
        branches = []
        for ifnode in (node.args.get("ifs") or []):
            branches.append({
                "when": _convert_expr(ifnode.this, exp),
                "then": _convert_expr(ifnode.args.get("true"), exp),
            })
        default = node.args.get("default")
        else_expr = _convert_expr(default, exp) if default is not None else None
        return {
            "type": "case",
            "base": base,
            "branches": branches,
            "else": else_expr,
            "text": _sql_text(node),
        }
    if isinstance(node, exp.Neg):
        return {
            "type": "unary",
            "op": "-",
            "expr": _convert_expr(node.this, exp),
            "text": _sql_text(node),
        }
    if isinstance(node, exp.Like):
        op = "NOT LIKE" if node.args.get("negated") else "LIKE"
        return {
            "type": "binary",
            "op": op,
            "left": _convert_expr(node.this, exp),
            "right": _convert_expr(node.expression, exp),
            "text": _sql_text(node),
        }
    if isinstance(node, exp.Binary):
        op = _BIN_OP_MAP.get(type(node).__name__, type(node).__name__)
        return {
            "type": "binary",
            "op": op,
            "left": _convert_expr(node.left, exp),
            "right": _convert_expr(node.right, exp),
            "text": _sql_text(node),
        }
    if isinstance(node, exp.Window):
        inner = node.this
        result = _convert_expr(inner, exp)
        full = _sql_text(node)
        if isinstance(result, dict):
            result["text"] = full
            if result.get("type") == "function":
                func_sql = _sql_text(inner)
                over_part = full[len(func_sql):].strip() if full.startswith(func_sql) else full
                if over_part.upper().startswith("OVER "):
                    over_part = over_part[5:].strip()
                result["window"] = over_part
        return result
    if isinstance(node, exp.Func):
        return _convert_function(node, exp)
    # fallback: treat unknown nodes as column-name text
    return {
        "type": "column",
        "qualifier": None,
        "name": _sql_text(node),
        "text": _sql_text(node),
    }


# ==================== Entry point ====================

def parse_sql(sql: str) -> Dict:
    """Prefer sqlglot; fall back to the built-in Parser on failure."""
    try:
        return parse_sql_via_sqlglot(sql)
    except Exception:
        return Parser(sql).parse_statement()


def build_config(sql: str, canvas: Tuple[int, int] = (1200, 620)) -> Tuple[Dict, Dict]:
    ast = parse_sql(sql)
    return ast, ConfigBuilder(ast, canvas).build()


def _parse_canvas(text: str) -> Tuple[int, int]:
    match = re.match(r"^\s*(\d+)\s*[xX*]\s*(\d+)\s*$", text or "")
    if not match:
        raise SystemExit("--canvas format should be 1200x620, got: %r" % text)
    return int(match.group(1)), int(match.group(2))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="SQL -> AST -> metaphor-diagram JSON config (nodes = relational entities, edges = clauses/operations)")
    parser.add_argument("--sql", help="SQL text directly")
    parser.add_argument("--sql-file", help="read from a .sql file", default="tool/examples/easy_example.sql")
    parser.add_argument("--out", help="output JSON path (prints to stdout by default)", default="default.json")
    parser.add_argument("--ast", action="store_true", help="print AST only, do not generate config")
    parser.add_argument("--canvas", default="1200x620", help="canvas size, e.g. 1200x620")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation")
    args = parser.parse_args(argv)

    if args.sql_file:
        sql = Path(args.sql_file).read_text(encoding="utf-8")
    elif args.sql:
        sql = args.sql
    else:
        parser.error("--sql or --sql-file is required")

    try:
        ast, config = build_config(sql, _parse_canvas(args.canvas))
    except ParseError as exc:
        raise SystemExit("SQL parse failed: %s" % exc)

    if args.ast:
        print(json.dumps(ast, ensure_ascii=False, indent=args.indent))
        return 0

    payload = json.dumps(config, ensure_ascii=False, indent=args.indent)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
        print("Generated %s (%d nodes, %d edges)"
              % (out_path, len(config["nodes"]), len(config["edges"])))
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




