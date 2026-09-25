// Backend-agnostic signal expression IR + a tiny DSL parser.
//
// A "signal" is a scalar expression evaluated per (company, month t) over that company's
// net_flow time series. This AST is the seam: the LLVM backend (jit_backtest.cpp) lowers it
// to native code today; an NVPTX/MLIR backend can lower the SAME tree to a GPU kernel later.
//
// Grammar (whitespace-insensitive):
//   expr   := term (('+' | '-') term)*
//   term   := factor (('*' | '/') factor)*
//   factor := number | 'nf' | 'lag' '(' int ')' | 'ewma' '(' int ',' number ')'
//           | 'mean' '(' int ')' | '(' expr ')' | '-' factor
// Primitives:  nf = net_flow[t];  lag(k) = net_flow[t-k];
//              ewma(k,d) = decay-d EWMA over the last k months;  mean(k) = mean over last k.
#pragma once
#include <string>
#include <memory>
#include <stdexcept>
#include <cctype>

namespace tf::sig {

enum class Op { Const, Nf, Lag, Ewma, Mean, Add, Sub, Mul, Div };

struct Node {
    Op op;
    double c = 0;   // Const value / Ewma decay
    int k = 0;      // Lag/Ewma/Mean window
    std::unique_ptr<Node> a, b;
    Node(Op o) : op(o) {}
};
using NodePtr = std::unique_ptr<Node>;

inline NodePtr num(double v){ auto n=std::make_unique<Node>(Op::Const); n->c=v; return n; }
inline NodePtr bin(Op o, NodePtr a, NodePtr b){ auto n=std::make_unique<Node>(o); n->a=std::move(a); n->b=std::move(b); return n; }

// --- recursive-descent parser ---
class Parser {
public:
    explicit Parser(std::string s) : src_(std::move(s)) {}
    NodePtr parse() {
        auto e = expr();
        skip();
        if (pos_ != src_.size()) throw std::runtime_error("trailing input at '" + src_.substr(pos_) + "'");
        return e;
    }
private:
    std::string src_; size_t pos_ = 0;
    void skip(){ while (pos_ < src_.size() && std::isspace((unsigned char)src_[pos_])) ++pos_; }
    bool eat(char c){ skip(); if (pos_ < src_.size() && src_[pos_]==c){ ++pos_; return true; } return false; }
    void expect(char c){ if(!eat(c)) throw std::runtime_error(std::string("expected '")+c+"'"); }
    bool peekWord(const char* w){ skip(); size_t n=std::string(w).size();
        return src_.compare(pos_, n, w)==0 && (pos_+n>=src_.size() || !std::isalnum((unsigned char)src_[pos_+n])); }
    void word(const char* w){ skip(); pos_ += std::string(w).size(); }
    double number(){ skip(); size_t s=pos_;
        while(pos_<src_.size() && (std::isdigit((unsigned char)src_[pos_])||src_[pos_]=='.'||src_[pos_]=='e'||src_[pos_]=='-'&&pos_==s)) ++pos_;
        if(pos_==s) throw std::runtime_error("expected number");
        return std::stod(src_.substr(s,pos_-s)); }
    int integer(){ return (int)number(); }

    NodePtr expr(){ auto a=term();
        for(;;){ skip(); if(eat('+')) a=bin(Op::Add,std::move(a),term());
                 else if(eat('-')) a=bin(Op::Sub,std::move(a),term()); else break; }
        return a; }
    NodePtr term(){ auto a=factor();
        for(;;){ skip(); if(eat('*')) a=bin(Op::Mul,std::move(a),factor());
                 else if(eat('/')) a=bin(Op::Div,std::move(a),factor()); else break; }
        return a; }
    NodePtr factor(){
        skip();
        if(eat('-')) return bin(Op::Sub, num(0.0), factor());
        if(eat('(')){ auto e=expr(); expect(')'); return e; }
        if(peekWord("nf")){ word("nf"); return std::make_unique<Node>(Op::Nf); }
        if(peekWord("lag")){ word("lag"); expect('('); int k=integer(); expect(')');
            auto n=std::make_unique<Node>(Op::Lag); n->k=k; return n; }
        if(peekWord("ewma")){ word("ewma"); expect('('); int k=integer(); expect(','); double d=number(); expect(')');
            auto n=std::make_unique<Node>(Op::Ewma); n->k=k; n->c=d; return n; }
        if(peekWord("mean")){ word("mean"); expect('('); int k=integer(); expect(')');
            auto n=std::make_unique<Node>(Op::Mean); n->k=k; return n; }
        return num(number());
    }
};

inline NodePtr parse(const std::string& s){ return Parser(s).parse(); }

} // namespace tf::sig
