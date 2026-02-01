import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import akshare as ak
from datetime import datetime
import glob
import hmac
import hashlib
import base64
import time
import urllib.parse
import json
import requests

def _get_env(key, default, cast_type=str):
    val = os.environ.get(key)
    if val is None: return default
    if cast_type == bool:
        return val.lower() in ('true', '1', 't', 'y', 'yes')
    return cast_type(val)

INDEX_CODE = _get_env('INDEX_CODE', '000001')
START_DATE = _get_env('START_DATE', '20240101') # 训练数据开始
END_DATE = _get_env('END_DATE', '20270101') # 训练数据结束
BATCH_SIZE = _get_env('BATCH_SIZE', 1024, int)
TRAIN_ITERATIONS = _get_env('TRAIN_ITERATIONS', 100, int)
MAX_SEQ_LEN = _get_env('MAX_SEQ_LEN', 10, int)
COST_RATE = _get_env('COST_RATE', 0.0004, float)
LAST_NDAYS = _get_env('LAST_NDAYS', 50, int)      # 用于展示最近交易日的数量
HOLD_PERIOD = _get_env('HOLD_PERIOD', 11, int)     # 持仓周期（包含买入当天后的第2..第HOLD_PERIOD天作为卖出候选）
FORCE_TRAIN = _get_env('FORCE_TRAIN', False, bool)  # 若为False且存在本地公式，则直接加载；若为True则强制重新训练
ONLY_LONG = _get_env('ONLY_LONG', True, bool)     # 是否仅做多，适配A股市场
BEST_FORMULA = _get_env('BEST_FORMULA', '')       # 环境变量公式
CODE_FORMULA = _get_env('CODE_FORMULA', '')       # 组合环境变量 (code:formula)

print("len of code_formula = " + str(len(CODE_FORMULA)))

# 解析 CODE_FORMULA
if CODE_FORMULA and ':' in CODE_FORMULA:
    parts = CODE_FORMULA.split(':', 1)
    INDEX_CODE = parts[0]
    BEST_FORMULA = parts[1]

DINGTALK_WEBHOOK = _get_env('DINGTALK_WEBHOOK', '')
DINGTALK_SECRET = _get_env('DINGTALK_SECRET', '')

def send_dingtalk_msg(text):
    if not DINGTALK_WEBHOOK:
        return
    
    url = DINGTALK_WEBHOOK
    if DINGTALK_SECRET:
        timestamp = str(round(time.time() * 1000))
        secret_enc = DINGTALK_SECRET.encode('utf-8')
        string_to_sign = '{}\n{}'.format(timestamp, DINGTALK_SECRET)
        string_to_sign_enc = string_to_sign.encode('utf-8')
        hmac_code = hmac.new(secret_enc, string_to_sign_enc, digestmod=hashlib.sha256).digest()
        sign = urllib.parse.quote(base64.b64encode(hmac_code))
        url = f"{DINGTALK_WEBHOOK}&timestamp={timestamp}&sign={sign}"

    headers = {'Content-Type': 'application/json'}
    data = {
        "msgtype": "text",
        "text": {
            "content": text
        }
    }
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(data), timeout=10)
        print(f"DingTalk notification sent, status: {resp.status_code}")
    except Exception as e:
        print(f"Failed to send DingTalk notification: {e}")

DATA_CACHE_PATH = INDEX_CODE + '_data_cache_final.parquet'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

@torch.jit.script
def _ts_delay(x: torch.Tensor, d: int) -> torch.Tensor:
    if d == 0: return x
    pad = torch.zeros((x.shape[0], d), device=x.device)
    return torch.cat([pad, x[:, :-d]], dim=1)

@torch.jit.script
def _op_gate(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mask = (condition > 0).float()
    return mask * x + (1.0 - mask) * y

@torch.jit.script
def _op_jump(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + 1e-6
    z = (x - mean) / std
    return torch.relu(z - 3.0)

@torch.jit.script
def _op_decay(x: torch.Tensor) -> torch.Tensor:
    return x + 0.8 * _ts_delay(x, 1) + 0.6 * _ts_delay(x, 2)

OPS_CONFIG = [
    ('ADD', lambda x, y: x + y, 2),
    ('SUB', lambda x, y: x - y, 2),
    ('MUL', lambda x, y: x * y, 2),
    ('DIV', lambda x, y: x / (y + 1e-6), 2),
    ('NEG', lambda x: -x, 1),
    ('ABS', torch.abs, 1),
    ('SIGN', torch.sign, 1),
    ('GATE', _op_gate, 3),
    ('JUMP', _op_jump, 1),
    ('DECAY', _op_decay, 1),
    ('DELAY1', lambda x: _ts_delay(x, 1), 1),
    ('MAX3', lambda x: torch.max(x, torch.max(_ts_delay(x,1), _ts_delay(x,2))), 1)
]

FEATURES = ['RET', 'RET5', 'VOL_CHG', 'V_RET', 'TREND', 'F_BUY_F_REPLAY']

VOCAB = FEATURES + [cfg[0] for cfg in OPS_CONFIG]
VOCAB_SIZE = len(VOCAB)
OP_FUNC_MAP = {i + len(FEATURES): cfg[1] for i, cfg in enumerate(OPS_CONFIG)}
OP_ARITY_MAP = {i + len(FEATURES): cfg[2] for i, cfg in enumerate(OPS_CONFIG)}

class AlphaGPT(nn.Module):
    def __init__(self, d_model=64, n_head=4, n_layer=2):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, MAX_SEQ_LEN + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=128, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layer, enable_nested_tensor=False)

        self.ln_f = nn.LayerNorm(d_model)
        self.head_actor = nn.Linear(d_model, VOCAB_SIZE)
        self.head_critic = nn.Linear(d_model, 1)

    def forward(self, idx):
        B, T = idx.size()
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(idx.device)
        x = self.blocks(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        last = x[:, -1, :]
        return self.head_actor(last), self.head_critic(last)

class DataEngine:
    def __init__(self):
        pass
    def load(self):
        print(f"Fetching {INDEX_CODE}...")

        df = ak.stock_zh_a_hist(symbol=INDEX_CODE, period="daily", start_date=START_DATE, end_date=END_DATE, adjust="qfq")
        if df is None or df.empty:
            try:
                df = ak.index_zh_a_hist(symbol=INDEX_CODE, period="daily", start_date=START_DATE, end_date=END_DATE)
            except:
                pass
        if df is None or df.empty:
            try:
                df = ak.fund_etf_hist_em(symbol=INDEX_CODE, period="daily", start_date=START_DATE, end_date=END_DATE, adjust="qfq")
            except:
                pass
        if df is None or df.empty:
            try:
                df = ak.fund_lof_hist_em(symbol=INDEX_CODE, period="daily", start_date=START_DATE, end_date=END_DATE, adjust="qfq")
            except:
                pass
        if df is None or df.empty:
            raise ValueError("未获取到数据，请检查接口调用或网络是否正常")

        df = df.sort_values('日期').reset_index(drop=True)
        # df.to_parquet(DATA_CACHE_PATH)

        for col in ['开盘', '最高', '最低', '收盘', '成交量']:
            df[col] = pd.to_numeric(df[col], errors='coerce').ffill().bfill()

        self.dates = pd.to_datetime(df['日期'])

        close = df['收盘'].values.astype(np.float32)
        open_ = df['开盘'].values.astype(np.float32)
        high = df['最高'].values.astype(np.float32)
        low = df['最低'].values.astype(np.float32)
        vol = df['成交量'].values.astype(np.float32)

        # 特征因子'RET'
        ret = np.zeros_like(close)
        ret[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-6)

        # 特征因子'RET5
        ret5 = pd.Series(close).pct_change(5).fillna(0).values.astype(np.float32)

        # 特征因子'VOL_CHG'
        vol_ma = pd.Series(vol).rolling(20).mean().values
        vol_chg = np.zeros_like(vol)
        mask = vol_ma > 0
        vol_chg[mask] = vol[mask] / vol_ma[mask] - 1
        vol_chg = np.nan_to_num(vol_chg).astype(np.float32)

        # 特征因子'V_RET'
        v_ret = (ret * (vol_chg + 1)).astype(np.float32)

        # 特征因子'TREND'
        ma60 = pd.Series(close).rolling(60).mean().values
        trend = np.zeros_like(close)
        mask = ma60 > 0
        trend[mask] = close[mask] / ma60[mask] - 1
        trend = np.nan_to_num(trend).astype(np.float32)

        # 特征因子'F_BUY_F_REPLAY'
        f_balance,f_buy,f_replay,s_balance = get_margin_balance(INDEX_CODE, pd.to_datetime(df['日期']).dt.strftime('%Y%m%d').tolist())
        f_buy_f_replay = f_buy - f_replay

        # 计算oto收益率
        # 可配置持仓周期：若 signal 表示持仓 1，则在 next-open 买入，
        # 在接下来的 HOLD_PERIOD 天内（含买入当天后的第2天..第HOLD_PERIOD天）
        # 优先选择第一个正收益的开盘价作为卖出，否则在第 HOLD_PERIOD 天卖出。
        open_tensor = torch.from_numpy(open_).to(DEVICE)
        open_t1 = torch.roll(open_tensor, -1)
        den = open_t1 + 1e-6
        ret_list = []
        for k in range(2, HOLD_PERIOD + 1):
            open_tk = torch.roll(open_tensor, -k)
            ret_k = (open_tk - open_t1) / den
            ret_list.append(ret_k)

        N = open_tensor.shape[0]
        h = HOLD_PERIOD
        if h < 2:
            raise ValueError("HOLD_PERIOD 必须大于等于为2")
        ret_mat = torch.full((h - 1, N), -float('inf'), device=DEVICE)
        for k in range(2, h + 1):
            valid_len = N - k
            if valid_len > 0:
                numer = open_tensor[k:]
                denom = open_tensor[1:1 + valid_len] + 1e-6
                arr = (numer - denom) / denom
                ret_mat[k - 2, :valid_len] = arr

        # 统一的选择逻辑：long/short 独立选择
        valid_mask = ret_mat != -float('inf')
        pos_mask = (ret_mat > 0) & valid_mask
        neg_mask = (ret_mat < 0) & valid_mask
        any_pos = pos_mask.any(dim=0)
        any_neg = neg_mask.any(dim=0)
        first_pos_idx = torch.argmax(pos_mask.int(), dim=0)
        first_neg_idx = torch.argmax(neg_mask.int(), dim=0)
        last_valid_idx = (valid_mask.sum(dim=0) - 1)
        has_valid = last_valid_idx >= 0

        indices = torch.arange(N, device=DEVICE)
        # long: 若存在正收益则取第一个正收益，否则取最后可用候选
        select_long_idx = torch.where(any_pos, first_pos_idx, last_valid_idx.clamp(min=0))
        selected_long = ret_mat[select_long_idx, indices]
        selected_long = torch.where(has_valid, selected_long, torch.zeros_like(selected_long))

        if ONLY_LONG:
            self.target_oto_ret = selected_long
        else:
            # short: 若存在负收益则取第一个负收益，否则取最后可用候选
            select_short_idx = torch.where(any_neg, first_neg_idx, last_valid_idx.clamp(min=0))
            selected_short = ret_mat[select_short_idx, indices]
            selected_short = torch.where(has_valid, selected_short, torch.zeros_like(selected_short))
            self.target_oto_ret_long = selected_long
            self.target_oto_ret_short = selected_short
            # 为兼容旧接口，保留 target_oto_ret 指向 long 的版本
            self.target_oto_ret = selected_long

        # Robust Normalization (确保返回的是 float32 的 numpy)
        def robust_norm(x):
            x = x.astype(np.float32) # 强制转类型
            median = np.nanmedian(x)
            mad = np.nanmedian(np.abs(x - median)) + 1e-6
            res = (x - median) / mad
            return np.clip(res, -5, 5).astype(np.float32)
        
        # 构建特征张量
        self.feat_data = torch.stack([
            torch.from_numpy(robust_norm(ret)).to(DEVICE),
            torch.from_numpy(robust_norm(ret5)).to(DEVICE),
            torch.from_numpy(robust_norm(vol_chg)).to(DEVICE),
            torch.from_numpy(robust_norm(v_ret)).to(DEVICE),
            torch.from_numpy(robust_norm(trend)).to(DEVICE),
            f_buy_f_replay
        ])

        self.raw_open = open_tensor
        self.raw_close = torch.from_numpy(close).to(DEVICE)
        self.split_idx = int(len(df) * 0.8)
        print(f"{INDEX_CODE} Data Ready. Normalization Fixed.")
        return self

class DeepQuantMiner:
    def __init__(self, engine):
        self.engine = engine
        self.model = AlphaGPT().to(DEVICE)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-5) # AdamW 防止过拟合
        self.best_sharpe = -10.0
        self.best_formula_tokens = None

    def get_strict_mask(self, open_slots, step):
        # 严格的 Action Masking，确保生成合法的 Polish Notation 树
        B = open_slots.shape[0]
        mask = torch.full((B, VOCAB_SIZE), float('-inf'), device=DEVICE)
        remaining_steps = MAX_SEQ_LEN - step

        done_mask = (open_slots == 0)
        mask[done_mask, 0] = 0.0 # Pad with first feature

        active_mask = ~done_mask
        # 如果剩余步数不够填坑了，必须选 Feature (arity=0)
        must_pick_feat = (open_slots >= remaining_steps)

        mask[active_mask, :len(FEATURES)] = 0.0
        can_pick_op_mask = active_mask & (~must_pick_feat)
        if can_pick_op_mask.any():
            mask[can_pick_op_mask, len(FEATURES):] = 0.0
        return mask

    def solve_one(self, tokens):
        stack = []
        try:
            # 倒序解析 (Reverse Polish like)
            for t in reversed(tokens):
                if t < len(FEATURES):
                    stack.append(self.engine.feat_data[t])
                else:
                    arity = OP_ARITY_MAP[t]
                    if len(stack) < arity: raise ValueError
                    args = [stack.pop() for _ in range(arity)]
                    func = OP_FUNC_MAP[t]
                    if arity == 2: res = func(args[0], args[1])
                    else: res = func(args[0])

                    if torch.isnan(res).any(): res = torch.nan_to_num(res)
                    stack.append(res)

            if len(stack) >= 1:
                final = stack[-1]
                # 过滤掉常数因子
                if final.std() < 1e-4: return None
                return final
        except:
            return None
        return None

    def solve_batch(self, token_seqs):
        B = token_seqs.shape[0]
        results = torch.zeros((B, self.engine.feat_data.shape[1]), device=DEVICE)
        valid_mask = torch.zeros(B, dtype=torch.bool, device=DEVICE)

        for i in range(B):
            res = self.solve_one(token_seqs[i].cpu().tolist())
            if res is not None:
                results[i] = res
                valid_mask[i] = True
        return results, valid_mask
    
    def backtest(self, factors):
        if factors.shape[0] == 0: return torch.tensor([], device=DEVICE)

        split = self.engine.split_idx
        # target_oto will be computed per-position below (to support long/short targets)
        rewards = torch.zeros(factors.shape[0], device=DEVICE)

        for i in range(factors.shape[0]):
            f = factors[i, :split]

            if torch.isnan(f).all() or (f == 0).all() or f.numel() == 0:
                rewards[i] = -2.0
                continue

            sig = torch.tanh(f)
            pos = torch.sign(sig)
            if ONLY_LONG:
                pos[pos == -1] = 0

            turnover = torch.abs(pos - torch.roll(pos, 1))
            if turnover.numel() > 0:
                turnover[0] = 0.0
            else:
                rewards[i] = -2.0
                continue

            # 根据仓位选择对应的 target_oto（long/short/flat）
            if ONLY_LONG:
                target_oto = self.engine.target_oto_ret[:split]
            else:
                long_t = self.engine.target_oto_ret_long[:split]
                short_t = self.engine.target_oto_ret_short[:split]
                target_oto = torch.where(pos == 1, long_t, torch.where(pos == -1, short_t, torch.zeros_like(long_t)))

            # 净收益
            pnl = pos * target_oto - turnover * COST_RATE

            try:
                pos_mask = (pos == 1)
                empty_mask = (pos == 0)
                neg_mask = (pos == -1)
                pos_count = int(pos_mask.sum().item())
                empty_count = int(empty_mask.sum().item())
                neg_count = int(neg_mask.sum().item())
                if int(pos_count / (pos_count + empty_count + neg_count) * 100) < 10:  # 多头仓位过少，惩罚
                    reward_score = 0.0
                else:
                    if ONLY_LONG:
                        win_count = int(((pos == 1) & (target_oto > 0)).sum().item())
                        win_rate_pct = (win_count / pos_count) * 100.0
                    else:
                        win_count = int(((pos == 1) & (target_oto > 0)).sum().item() + ((pos == -1) & (target_oto < 0)).sum().item())
                        win_rate_pct = (win_count / (pos_count + neg_count)) * 100.0
                    
                    # pnl 的平均收益（乘以100转为百分比）
                    avg_pnl = pnl.mean().item() * 100.0
                    sum_pnl = pnl.sum().item() * 100.0
                    
                    # 综合评分：胜率 × 平均收益
                    reward_score = win_rate_pct * avg_pnl
            except Exception:
                reward_score = 0.0

            rewards[i] = torch.tensor(float(reward_score), dtype=torch.float32, device=DEVICE)

        return rewards
    
    def find_best_formula_file(self):
        """查找最新的公式文件"""
        pattern = f"{INDEX_CODE}_best_formula_*.txt"
        files = glob.glob(pattern)
        if files:
            return max(files, key=os.path.getctime)  # 返回最新修改的文件
        return None
    
    def load_formula_from_file(self, filepath):
        """从文件加载公式"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # 解析格式: 第一行isSortino值，第二行开始是token序列
                lines = content.split('\n')
                best_sharpe = float(lines[0].split(':')[1].strip())
                tokens_str = lines[1].split(':')[1].strip()
                best_formula_tokens = [int(x) for x in tokens_str.strip('[]').split(',')]
                self.best_sharpe = best_sharpe
                self.best_formula_tokens = best_formula_tokens
                print(f"加载本地公式: {filepath}")
                print(f"   BestSortino: {best_sharpe:.3f}")
                return True
        except Exception as e:
            print(f"加载公式失败: {e}")
            return False
    
    def train(self):
        # 优先选择环境变量中的公式
        if BEST_FORMULA:
            encoded_tokens = self.encode(BEST_FORMULA)
            if encoded_tokens:
                self.best_formula_tokens = encoded_tokens
                # 计算初始得分
                f_val = self.solve_one(encoded_tokens)
                if f_val is not None:
                    self.best_sharpe = self.backtest(f_val.unsqueeze(0))[0].item()
                print(f"解析环境变量公式: {BEST_FORMULA}")
                print(f"   BestSortino: {self.best_sharpe:.3f}")
                return
        else:
            print("没有提供公式，退出")
            return

        # 检查是否需要加载本地公式
        if not FORCE_TRAIN:
            formula_file = self.find_best_formula_file()
            if formula_file:
                if self.load_formula_from_file(formula_file):
                    print(f"跳过训练，使用本地公式")
                    return
        
        print(f"Training for Stable Profit... MAX_LEN={MAX_SEQ_LEN}")
        pbar = tqdm(range(TRAIN_ITERATIONS))

        for _ in pbar:
            # 1. Generate
            B = BATCH_SIZE
            open_slots = torch.ones(B, dtype=torch.long, device=DEVICE)
            log_probs, tokens = [], []
            curr_inp = torch.zeros((B, 1), dtype=torch.long, device=DEVICE)

            for step in range(MAX_SEQ_LEN):
                logits, val = self.model(curr_inp)
                mask = self.get_strict_mask(open_slots, step)
                dist = Categorical(logits=(logits + mask))
                action = dist.sample()

                log_probs.append(dist.log_prob(action))
                tokens.append(action)
                curr_inp = torch.cat([curr_inp, action.unsqueeze(1)], dim=1)

                is_op = action >= len(FEATURES)
                delta = torch.full((B,), -1, device=DEVICE)
                arity_tens = torch.zeros(VOCAB_SIZE, dtype=torch.long, device=DEVICE)
                for k,v in OP_ARITY_MAP.items(): arity_tens[k] = v
                op_delta = arity_tens[action] - 1
                delta = torch.where(is_op, op_delta, delta)
                delta[open_slots==0] = 0
                open_slots += delta

            seqs = torch.stack(tokens, dim=1)

            # 2. Evaluate
            with torch.no_grad():
                f_vals, valid_mask = self.solve_batch(seqs)
                valid_idx = torch.where(valid_mask)[0]
                rewards = torch.full((B,), -1.0, device=DEVICE) # 默认惩罚

                if len(valid_idx) > 0:
                    bt_scores = self.backtest(f_vals[valid_idx])
                    rewards[valid_idx] = bt_scores

                    best_sub_idx = torch.argmax(bt_scores)
                    current_best_score = bt_scores[best_sub_idx].item()

                    if current_best_score > self.best_sharpe:
                        self.best_sharpe = current_best_score
                        self.best_formula_tokens = seqs[valid_idx[best_sub_idx]].cpu().tolist()

            # 3. Update
            adv = rewards - rewards.mean()
            loss = -(torch.stack(log_probs, 1).sum(1) * adv).mean()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            pbar.set_postfix({'Valid': f"{len(valid_idx)/B:.1%}", 'BestSortino': f"{self.best_sharpe:.3f}"})
        
        # 训练完成后保存公式
        self.save_formula()

    def save_formula(self):
        """保存最佳公式到文件"""
        if self.best_formula_tokens is None:
            print("没有有效的公式可保存")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{INDEX_CODE}_best_formula_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"BestSortino: {self.best_sharpe:.4f}\n")
            f.write(f"Tokens: {self.best_formula_tokens}\n")
            f.write(f"Formula: {self.decode()}\n")
        
        print(f"公式已保存: {filename}")


    def decode(self, tokens=None):
        if tokens is None: tokens = self.best_formula_tokens
        if tokens is None: return "N/A"
        stream = list(tokens)
        def _parse():
            if not stream: return ""
            t = stream.pop(0)
            if t < len(FEATURES): return FEATURES[t]
            args = [_parse() for _ in range(OP_ARITY_MAP[t])]
            return f"{VOCAB[t]}({','.join(args)})"
        try: return _parse()
        except: return "Invalid"

    def encode(self, formula_str):
        """将公式字符串转换为token序列"""
        import re
        # 词法分析：提取特征、操作符、括号和逗号
        tokens_raw = re.findall(r'[A-Z0-9_]+|\(|\)|,', formula_str)
        
        vocab_map = {name: i for i, name in enumerate(VOCAB)}
        feat_set = set(FEATURES)
        
        pos = 0
        def _parse():
            nonlocal pos
            if pos >= len(tokens_raw):
                return []
            
            token = tokens_raw[pos]
            pos += 1
            
            if token in feat_set:
                return [vocab_map[token]]
            elif token in vocab_map: # 这是一个操作符
                op_idx = vocab_map[token]
                arity = OP_ARITY_MAP[op_idx]
                
                # 下一个应该是 '('
                if pos < len(tokens_raw) and tokens_raw[pos] == '(':
                    pos += 1 # 跳过 '('
                    
                    args_tokens = []
                    for i in range(arity):
                        args_tokens.extend(_parse())
                        if i < arity - 1:
                            if pos < len(tokens_raw) and tokens_raw[pos] == ',':
                                pos += 1 # 跳过 ','
                    
                    # 最后一个应该是 ')'
                    if pos < len(tokens_raw) and tokens_raw[pos] == ')':
                        pos += 1 # 跳过 ')'
                        
                    return [op_idx] + args_tokens
            return []

        return _parse()

def final_reality_check(miner, engine):
    print("\n" + "="*60)
    print("FINAL REALITY CHECK (Out-of-Sample)")
    print("="*60)

    formula_str = miner.decode()
    if miner.best_formula_tokens is None: 
        return None
    print(f"Strategy Formula: {formula_str}")

    # 1. 获取全量因子值
    factor_all = miner.solve_one(miner.best_formula_tokens)
    if factor_all is None: return

    # 2. 提取测试集数据 (Strict OOS)
    split = engine.split_idx
    test_dates = engine.dates[split:]
    test_factors = factor_all[split:].cpu().numpy()

    # 使用 Open-to-Open 收益
    # test_ret 将根据仓位在后面计算（支持 long/short）

    # 减少噪音
    rolling_mean_factor = pd.Series(test_factors).rolling(3).mean().fillna(0).values
    signal = np.tanh(test_factors)

    # 仓位
    position = np.sign(signal)
    if ONLY_LONG:
        position[position == -1] = 0

    # 根据仓位选择对应的 test_ret（支持 long/short）
    if ONLY_LONG:
        test_ret = engine.target_oto_ret[split:].cpu().numpy()
    else:
        long_t = engine.target_oto_ret_long[split:].cpu().numpy()
        short_t = engine.target_oto_ret_short[split:].cpu().numpy()
        test_ret = np.where(position == 1, long_t, np.where(position == -1, short_t, np.zeros_like(long_t)))

    # 检查涨跌停/停牌 (Limit Move Check)
    # 模拟：如果 next_open 相对于 close 涨跌幅超过 9.5%，则无法成交
    # raw_close[t], raw_open[t+1]
    # 我们检查 t+1 开盘是否可交易。

    raw_close = engine.raw_close[split:].cpu().numpy()
    raw_open_next = engine.raw_open[split:].cpu().numpy() # 这里稍微错位，简化处理
    # 实际上，DataEngine需要更精细的时间对齐来做Limit Check，这里做个简单近似

    # 换手
    turnover = np.abs(position - np.roll(position, 1))
    turnover[0] = 0

    # PnL
    daily_ret = position * test_ret - turnover * COST_RATE

    # 4. 统计
    equity = (1 + daily_ret).cumprod()

    total_ret = equity[-1] - 1
    ann_ret = equity[-1] ** (252/len(equity)) - 1
    vol = np.std(daily_ret) * np.sqrt(252)
    sharpe = (ann_ret - 0.02) / (vol + 1e-6)

    # Max Drawdown
    dd = 1 - equity / np.maximum.accumulate(equity)
    max_dd = np.max(dd)
    calmar = ann_ret / (max_dd + 1e-6)

    print(f"Test Period    : {test_dates.iloc[0].date()} ~ {test_dates.iloc[-1].date()}")
    print(f"Total Return   : {total_ret:.2%}")
    print(f"Ann. Return    : {ann_ret:.2%}")
    print(f"Ann. Volatility: {vol:.2%}")
    print(f"Sharpe Ratio   : {sharpe:.3f}")
    print(f"Max Drawdown   : {max_dd:.2%}")
    print(f"Calmar Ratio   : {calmar:.3f}")

    try:
        if ONLY_LONG:
            pos_mask = (position == 1)
            total_positions = int(np.sum(pos_mask))
            if total_positions > 0:
                success_count = int(np.sum((pos_mask) & (test_ret > 0)))
                success_rate = success_count / total_positions
            else:
                success_count = 0
                success_rate = 0.0
        else:
            pos_mask = (position == 1)
            neg_mask = (position == -1)
            total_positions = int(np.sum(pos_mask) + np.sum(neg_mask))
            if total_positions > 0:
                success_count = int(np.sum((pos_mask) & (test_ret > 0)) + np.sum((neg_mask) & (test_ret < 0)))
                success_rate = success_count / total_positions
            else:
                success_count = 0
                success_rate = 0.0
        print(f"Prediction Success: {success_count}/{total_positions} = {success_rate:.1%}")
    except Exception:
        print("Prediction Success: N/A")

    print("-" * 60)
    
    # 同时保存性能指标到文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_filename = f"{INDEX_CODE}_metrics_{timestamp}.txt"
    
    metrics_info = f"""Strategy Formula: {formula_str}
                    BestSortino: {miner.best_sharpe:.4f}
                    ------------------------------------------------------------
                    Test Period    : {test_dates.iloc[0].date()} ~ {test_dates.iloc[-1].date()}
                    Ann. Return    : {ann_ret:.2%}
                    Ann. Volatility: {vol:.2%}
                    Sharpe Ratio   : {sharpe:.3f}
                    Max Drawdown   : {max_dd:.2%}
                    Calmar Ratio   : {calmar:.3f}"""
    
    with open(metrics_filename, 'w', encoding='utf-8') as f:
        f.write(metrics_info)
    
    print(f"性能指标已保存: {metrics_filename}")

    # 5. Plot
    plt.style.use('bmh')
    plt.figure(figsize=(12, 6))

    # 绘制策略曲线
    plt.plot(test_dates, equity, label='Strategy (Open-to-Open)', linewidth=1.5)

    # 绘制基准 (Buy & Hold)
    # 基准也应该是 Open-to-Open
    bench_ret = test_ret
    bench_equity = (1 + bench_ret).cumprod()
    plt.plot(test_dates, bench_equity, label='Benchmark (CSI 300)', alpha=0.5, linewidth=1)
    
    plt.title(f'Strict OOS Backtest: Ann Ret {ann_ret:.1%} | Sharpe {sharpe:.3f}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('strategy_performance.png')
    print("Chart saved to 'strategy_performance.png'")

def show_latest_positions(miner, engine, n_days=5):
    """输出最近n个交易日的position信息、收益率以及后续两个交易日的开盘价"""
    output_lines = []
    
    def log_print(msg):
        print(msg)
        output_lines.append(msg)

    log_print("\n" + "="*60)
    log_print(f"Latest {n_days} Trading Days Position Info")
    log_print("="*60)
    
    if miner.best_formula_tokens is None:
        log_print("No valid formula available")
        return
    
    # 1. 计算全量因子值
    factor_all = miner.solve_one(miner.best_formula_tokens)
    if factor_all is None:
        log_print("Failed to compute factors")
        return
    
    # 2. 提取测试集数据
    split = engine.split_idx
    test_dates = engine.dates[split:]
    test_factors = factor_all[split:].cpu().numpy()
    # test_ret will be computed after position is known (supports long/short)
    
    # 获取全量开盘价
    all_open = engine.raw_open.cpu().numpy()
    
    # 3. 计算signal和position
    signal = np.tanh(test_factors)
    position = np.sign(signal)
    if ONLY_LONG:
        position[position == -1] = 0  # 转换为long-only

    # 根据仓位选择 test_ret（支持 long/short）
    if ONLY_LONG:
        test_ret = engine.target_oto_ret[split:].cpu().numpy()
    else:
        long_t = engine.target_oto_ret_long[split:].cpu().numpy()
        short_t = engine.target_oto_ret_short[split:].cpu().numpy()
        test_ret = np.where(position == 1, long_t, np.where(position == -1, short_t, np.zeros_like(long_t)))

    # 换手（用于复合回报计算）
    turnover = np.abs(position - np.roll(position, 1))
    turnover[0] = 0
    
    # 4. 获取最后n天数据（或全部如果少于n天）
    n_display = min(n_days, len(test_dates))
    start_idx = len(test_dates) - n_display
    
    log_print(f"\n{'Date':<12} {'Position':<10} {'Return':<12} {'D1_Open':<12} {'ExitOff':<8} {'ExitOpen':<9}")
    log_print("-" * 72)
    
    # 用于统计总回报率和投资次数
    simple_sum_return = 0.0
    compound_equity = 1.0
    valid_days = 0
    investment_count = 0  # position=1的次数
    profit_count = 0      # position=1且收益>0的次数
    
    for i in range(start_idx, len(test_dates)):
        date_str = test_dates.iloc[i].strftime('%Y-%m-%d')
        pos_value = position[i]
        # 预先定义用于后续退出复现的变量，避免作用域未定义错误
        t1 = None
        chosen_offset = None
        
        # 检查是否有对应的future return
        if i < len(test_ret):
            ret_value = test_ret[i]
            ret_str = f"{ret_value:.2%}"

            full_idx = split + i

            # 逐项复现 t2-t1 .. t{HOLD_PERIOD}-t1 的计算以便对比（HOLD_PERIOD 为模块级变量）
            opens = []
            for k in range(1, HOLD_PERIOD + 1):
                idx_k = full_idx + k
                opens.append(all_open[idx_k] if idx_k < len(all_open) else None)

            # t1 为 opens[0]，候选为 opens[1]..opens[HOLD_PERIOD-1]
            chosen_ret = None
            chosen_offset = None
            t1 = opens[0]
            r_list = []
            if t1 is not None and t1 != 0:
                for p in range(1, HOLD_PERIOD):
                    ok = opens[p]
                    if ok is None:
                        r = None
                    else:
                        r = (ok - t1) / (t1 + 1e-6)
                    r_list.append(r)

                # 根据仓位优先选择第一个符合盈利条件的天：
                # - 如果 pos_value == 1（多头），选择第一个 r > 0
                # - 如果 pos_value == -1（空头），选择第一个 r < 0
                # 否则（无仓位或无法判断）不提前退出，回退到周期最后一天
                if pos_value == 1:
                    for idx_r, r in enumerate(r_list):
                        if r is not None and r > 0:
                            chosen_ret = r
                            chosen_offset = idx_r + 2
                            break
                elif pos_value == -1:
                    for idx_r, r in enumerate(r_list):
                        if r is not None and r < 0:
                            chosen_ret = r
                            chosen_offset = idx_r + 2
                            break

                if chosen_ret is None:
                    # 取 t{HOLD_PERIOD}-t1（周期最后一天）
                    chosen_ret = r_list[-1]
                    chosen_offset = HOLD_PERIOD

            # 简单回报之和（不含手续费）
            simple_sum_return += pos_value * ret_value
            # 复合回报计入换手手续费（按日累乘）
            daily_effective = pos_value * ret_value - turnover[i] * COST_RATE
            compound_equity *= (1.0 + daily_effective)
            valid_days += 1

            # 统计投资次数和盈利次数
            if pos_value == 1:
                investment_count += 1
                if ret_value > 0:
                    profit_count += 1
            if pos_value == -1:
                investment_count += 1
                if ret_value < 0:
                    profit_count += 1
        else:
            ret_str = "N/A"
        
        # 获取后续一个交易日的开盘价（对应在全量数据中的位置）
        full_idx = split + i
        d1_open = "N/A"
        if full_idx + 1 < len(all_open):
            d1_open = f"{all_open[full_idx + 1]:.3f}"
        
        # 计算退出信息（统一逻辑，优先选择第一个正收益，否则取最后一个有效候选）
        exit_offset = 'N/A'
        exit_open = 'N/A'
        if t1 is not None and t1 != 0:
            # 已在上文计算 r_list、chosen_ret、chosen_offset
            if 'chosen_offset' in locals() and chosen_offset is not None:
                exit_offset = chosen_offset
                exit_idx = full_idx + chosen_offset
                exit_open = f"{all_open[exit_idx]:.3f}" if exit_idx < len(all_open) else 'N/A'
            else:
                # 回退策略：如果没有选到，则取最后一个可用
                exit_offset = HOLD_PERIOD
                exit_idx = full_idx + HOLD_PERIOD
                exit_open = f"{all_open[exit_idx]:.3f}" if exit_idx < len(all_open) else 'N/A'

        log_print(f"{date_str:<12} {pos_value:<10.0f} {ret_str:<12} {d1_open:<13} {exit_offset:<8} {exit_open:<9}")

    log_print("-" * 72)
    if valid_days > 0 and investment_count > 0:
        win_rate = profit_count / investment_count
        log_print(f"Summary over these {valid_days} days:")
        log_print(f"  Investment Count: {investment_count}")
        log_print(f"  Profit Count    : {profit_count}")
        log_print(f"  Win Rate        : {win_rate:.2%}")
        log_print(f"  Simple Return   : {simple_sum_return:.2%}")
        log_print(f"  Compound Total  : {(compound_equity - 1):.2%}")
    else:
        log_print("No active trades in the selected period.")
    log_print("="*60 + "\n")

    # 发送钉钉消息
    if DINGTALK_WEBHOOK:
        full_msg = f"AlphaGPT Strategy [{INDEX_CODE}]\n" + "\n".join(output_lines)
        send_dingtalk_msg(full_msg)


def get_margin_balance(stock_code, date_list):
    """
    获取指定股票在时间周期内的两融余额数据，返回张量
    支持本地缓存和增量更新（按日期缓存）
    
    Args:
        stock_code: 股票代码，如 '002466'
        date_list: 日期列表，格式为 'YYYYMMDD'，从df['日期']提取
    
    Returns:
        tuple: 四个张量 (融资余额, 融资买入额, 融资偿还额, 融券余量)
               每个张量长度为date_list的长度
               如果该日期无数据，该值默认为0
    """
    # 创建本地缓存目录
    cache_dir = "margin_balance"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # 初始化数据字典：日期 -> 数据行
    margin_data = {}
    missing_dates = []
    
    # 检查本地缓存中的日期
    for date in date_list:
        cache_file = os.path.join(cache_dir, f"{date}_margin_data.parquet")
        
        if os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                # 从该日期的文件中过滤目标股票
                target_rows = df[df['标的证券代码'] == stock_code]
                if not target_rows.empty:
                    margin_data[date] = target_rows.iloc[0].to_dict()
            except Exception as e:
                print(f"Failed to load cache for {date}: {e}")
                missing_dates.append(date)
        else:
            missing_dates.append(date)
    
    # 获取缺失的日期数据
    if missing_dates:
        print(f"Checking trading days and fetching margin data for {len(missing_dates)} missing dates...")
        new_data_dict = _fetch_margin_data_from_api(stock_code, missing_dates, cache_dir)
        margin_data.update(new_data_dict)
    else:
        print(f"Using cached data for {stock_code} ({date_list[0]} ~ {date_list[-1]})")
    
    # 构建张量：为每个日期填充数据（无数据则为0）
    financing_balance = []
    financing_buy = []
    financing_repay = []
    short_balance = []
    
    for date in date_list:
        if date in margin_data:
            row = margin_data[date]
            financing_balance.append(float(row.get('融资余额', 0)))
            financing_buy.append(float(row.get('融资买入额', 0)))
            financing_repay.append(float(row.get('融资偿还额', 0)))
            short_balance.append(float(row.get('融券余量', 0)))
        else:
            # 无数据则默认为0
            financing_balance.append(0.0)
            financing_buy.append(0.0)
            financing_repay.append(0.0)
            short_balance.append(0.0)
    
    # 转换为张量
    financing_balance_tensor = torch.tensor(financing_balance, dtype=torch.float32, device=DEVICE)
    financing_buy_tensor = torch.tensor(financing_buy, dtype=torch.float32, device=DEVICE)
    financing_repay_tensor = torch.tensor(financing_repay, dtype=torch.float32, device=DEVICE)
    short_balance_tensor = torch.tensor(short_balance, dtype=torch.float32, device=DEVICE)

    print(f"Successfully processed {len(margin_data)} trading days for {stock_code}")
    print(f"Tensor lengths: {len(financing_balance)}")
    print(f"Margin Data - First Day ({date_list[0]}): F_Balance={financing_balance[0]:.0f}, S_Balance={short_balance[0]:.0f}")
    print(f"Margin Data - Last Day  ({date_list[-1]}): F_Balance={financing_balance[-1]:.0f}, S_Balance={short_balance[-1]:.0f}")

    return financing_balance_tensor, financing_buy_tensor, financing_repay_tensor, short_balance_tensor

def _fetch_margin_data_from_api(stock_code, date_list, cache_dir):
    """
    从akshare API获取两融数据，按日期保存到本地
    两阶段处理：先过滤交易日，再获取数据
    
    Args:
        stock_code: 股票代码
        date_list: 日期列表（格式 'YYYYMMDD'）
        cache_dir: 缓存目录
    
    Returns:
        dict: 日期 -> 数据字典（仅包含目标股票）
    """
    margin_data = {}
    failed_dates = []
    trading_dates = date_list
    for date in tqdm(trading_dates, desc="Fetching margin data"):
        try:
            # 调用API获取该日期的两融数据
            df = ak.stock_margin_detail_sse(date=date)
            
            if df is None or df.empty:
                failed_dates.append(date)
                continue
            
            # 保存整个日期的所有数据到本地
            cache_file = os.path.join(cache_dir, f"{date}_margin_data.parquet")
            df.to_parquet(cache_file)
            
            # 提取目标股票的数据
            target_rows = df[df['标的证券代码'] == stock_code]
            if not target_rows.empty:
                row = target_rows.iloc[0]
                margin_data[date] = {
                    '信用交易日期': date,
                    '标的证券代码': stock_code,
                    '融资余额': float(row.get('融资余额', 0)),
                    '融资买入额': float(row.get('融资买入额', 0)),
                    '融资偿还额': float(row.get('融资偿还额', 0)),
                    '融券余量': float(row.get('融券余量', 0))
                }
        except Exception as e:
            failed_dates.append(date)
            continue
    
    if failed_dates:
        print(f"Failed to fetch data for {len(failed_dates)} dates")
    
    return margin_data

if __name__ == "__main__":
    eng = DataEngine()
    eng.load()
    miner = DeepQuantMiner(eng)
    miner.train()
    # final_reality_check(miner, eng)
    show_latest_positions(miner, eng, n_days=LAST_NDAYS)