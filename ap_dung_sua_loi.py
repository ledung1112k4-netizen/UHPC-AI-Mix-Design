"""Áp dụng sửa lỗi giới hạn và ẩn giao diện cho app UHPC AI.

Chạy trong thư mục dự án: python ap_dung_sua_loi.py
Dừng AN TOÀN nếu cấu trúc mã thay đổi, chỉ ghi sau khi kiểm tra cú pháp.
Không xóa mô hình/CSV và không tự push GitHub.
"""
from __future__ import annotations
import ast
from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
PROJECT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
APP = PROJECT / 'app.py'
CORE = PROJECT / 'uhpc_core.py'
MARK = '# UHPC_ENGINEERING_GUARDS_V1'


def insert_after_assignment(source: str, function_name: str, variable: str, code: str, *, nested_function: str | None = None) -> str:
    tree = ast.parse(source)
    funcs = [v for v in tree.body if isinstance(v, (ast.FunctionDef, ast.AsyncFunctionDef)) and v.name == function_name]
    if len(funcs) != 1:
        raise RuntimeError(f"Không tìm được đúng 1 hàm {function_name}; dừng, không sửa.")
    func = funcs[0]
    if nested_function:
        funcs = [x for x in ast.walk(func) if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name == nested_function]
        if len(funcs) != 1:
            raise RuntimeError(f"Không tìm được đúng 1 hàm {nested_function} trong {function_name}.")
        func = funcs[0]
    matches = []
    for node in ast.walk(func):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == variable for t in targets):
                matches.append(node)
    if len(matches) != 1:
        raise RuntimeError(f"Không nhận diện được phép gán '{variable}' (thấy {len(matches)}) trong {function_name}.")
    node = matches[0]
    lines = source.splitlines(keepends=True)
    indent = lines[node.lineno - 1][:len(lines[node.lineno - 1]) - len(lines[node.lineno - 1].lstrip())]
    payload = ''.join(indent + l + '\n' for l in code.strip().splitlines())
    lines.insert(node.end_lineno, payload)
    patched = ''.join(lines)
    ast.parse(patched)
    return patched


def update_core(s: str) -> str:
    if MARK in s:
        raise RuntimeError('uhpc_core.py đã có bộ kiểm soát này; không cài hai lần.')
    if 'def optimize_mix(' not in s or 'def material_search_bounds(' not in s:
        raise RuntimeError('Không tìm thấy hàm tối ưu cần nâng cấp trong uhpc_core.py')
    # Dừng nếu optimizer không còn cùng các biến trung gian như bản đã kiểm chứng.
    parsed = ast.parse(s)
    opt = next((n for n in parsed.body if isinstance(n, ast.FunctionDef) and n.name == 'optimize_mix'), None)
    objective = next((n for n in ast.walk(opt) if isinstance(n, ast.FunctionDef) and n.name == 'objective'), None)
    if objective is None or not any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                                    and n.id == 'mix_arr' for n in ast.walk(objective)):
        raise RuntimeError('Hàm objective không dùng mix_arr như bản đã kiểm chứng; dừng an toàn.')
    # Sửa toàn bộ các tiêu chí chấp nhận trong bài toán NGƯỢC.
    s = insert_after_assignment(s, 'optimize_mix', 'bds',
        'bds = uhpc_eng.tighten_bounds(bds, MATERIALS, float(target_mpa))')
    s = insert_after_assignment(s, 'optimize_mix', 'loss',
        'eng_feasible, eng_penalty = uhpc_eng.assess_population(mix_arr, MATERIALS, float(target_mpa))\nloss = np.asarray(loss, dtype=float) + eng_penalty',
        nested_function='objective')
    s = insert_after_assignment(s, 'optimize_mix', 'feasible',
        'feasible = np.asarray(feasible, dtype=bool) & eng_feasible',
        nested_function='objective')
    # Cuối quá trình vẫn kiểm tra lại nghiệm: mất cờ này sẽ có nguy cơ hiển thị
    # phương án bất khả thi như một cấp phối hợp lệ.
    tree = ast.parse(s)
    opt = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'optimize_mix')
    check_vars = [name for name in ('physical_checks', 'checks') if any(
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
        for n in ast.walk(opt) if not isinstance(n, ast.FunctionDef) or n is opt)]
    if len(check_vars) != 1:
        raise RuntimeError('Không xác định được bảng checks đầu ra để bảo đảm không duyệt nghiệm sai.')
    checks_name = check_vars[0]
    s = insert_after_assignment(s, 'optimize_mix', checks_name,
        f'{checks_name}.update(uhpc_eng.mix_checks(proposal, float(target_mpa)))')
    if 'from scipy.optimize import differential_evolution\n' not in s:
        raise RuntimeError('Không tìm được vị trí thêm module trong uhpc_core.py')
    s = s.replace('from scipy.optimize import differential_evolution\n',
                  'from scipy.optimize import differential_evolution\nimport uhpc_engineering as uhpc_eng  ' + MARK + '\n', 1)
    ast.parse(s)
    return s


def update_app(s: str) -> str:
    if 'UHPC_HIDDEN_LIMITS_2026_10_28' in s:
        # Nếu đã ẩn trước đó, chỉ thêm lớp kiểm tra kết quả.
        pass
    else:
        # Gói ẩn UI đã kiểm thử với cấu trúc repo ứng với ảnh hiện tại.
        from hide_material_limits import rewrite
        s = rewrite(s)

    # Chặn bài toán thuận bằng biên thiết kế nội bộ chứ không để AI tự kết luận.
    if 'from uhpc_core import (' not in s:
        raise RuntimeError('Không tìm thấy import của uhpc_core')
    s = s.replace('from uhpc_core import (', 'from uhpc_engineering import forward_issues\nfrom uhpc_core import (', 1)
    marker = '    if submit_pred and any(not (lo <= mix[k] <= hi) for k, (lo, hi) in forward_limits.items()):\n'
    if marker not in s:
        raise RuntimeError('Không tìm thấy bước xác nhận đầu vào bài toán thuận sau khi ẩn biên.')
    s = s.replace(marker,
        '    input_issues = forward_issues(mix) if submit_pred else []\n'
        '    if submit_pred and input_issues:\n'
        '        st.session_state.pop("forward_result", None)\n'
        '        st.error("Cấp phối không đạt bộ kiểm tra kỹ thuật nội bộ: " + "; ".join(input_issues))\n'
        '    elif submit_pred and any(not (lo <= mix[k] <= hi) for k, (lo, hi) in forward_limits.items()):\n', 1)

    # Không xuất bảng cấp phối từ ứng viên chẩn đoán khi tất cả ràng buộc chưa đạt.
    old = ('        st.markdown("#### Bảng cấp phối đề xuất / ứng viên")\n'
           '        display_mix(result["mix"], batch_mode, batch_amount, "de_xuat")\n')
    if old not in s:
        raise RuntimeError('Không tìm thấy đoạn hiển thị bảng cấp phối ngược; dừng an toàn.')
    new = ('        if result["feasible"]:\n'
           '            st.markdown("#### Cấp phối đề xuất (sàng lọc AI, chưa thí nghiệm)")\n'
           '            display_mix(result["mix"], batch_mode, batch_amount, "de_xuat")\n'
           '        else:\n'
           '            st.warning("Không xuất cấp phối: ứng viên chưa thỏa ràng buộc kỹ thuật hoặc cường độ. ")\n')
    s = s.replace(old, new, 1)
    # Khi thay mục tiêu phải xóa cache kết quả inverse (bản hide đã gắn khóa đầy đủ).
    if 'st.session_state["inverse_choice_key"]' not in s:
        raise RuntimeError('Không tìm thấy quản lý trạng thái kết quả bài toán ngược.')
    ast.parse(s)
    if ('Biên tối thiểu – tối đa của từng vật liệu' in s
            or 'key="inverse_low_"' in s or 'key="forward_low_"' in s):
        raise RuntimeError('Bản vá chưa ẩn hết ô giới hạn vật liệu; dừng an toàn.')
    return s


def main():
    if not (APP.exists() and CORE.exists()):
        raise SystemExit('Cần chạy trong thư mục GitHub chứa app.py và uhpc_core.py. ' +
                         'Ví dụ: python ap_dung_sua_loi.py "D:\\UHPC-AI-Mix-Design"')
    outputs = {
        APP: update_app(APP.read_text(encoding='utf-8')),
        CORE: update_core(CORE.read_text(encoding='utf-8')),
    }
    # Kiểm tra hoàn tất và biên cảnh báo trước khi ghi bất kỳ byte nào.
    for path, content in outputs.items():
        compile(content, str(path), 'exec')
    # Không ghi đè bản sao lưu cũ; tạo hậu tố theo lần chạy.
    backs = []
    try:
        for path in outputs:
            backup = path.with_name(path.name + '.before_engineering_fix.bak')
            if backup.exists():
                raise RuntimeError(f'Bản sao lưu đã tồn tại: {backup}. Chưa sửa gì.')
            backs.append((path, backup))
        for path, backup in backs:
            shutil.copy2(path, backup)
        for path, content in outputs.items():
            path.write_text(content, encoding='utf-8')
        print('ĐÃ CẬP NHẬT app.py và uhpc_core.py.')
        print('Không xóa tệp mô hình, CSV hay module hiện có.')
        print('Đã lưu bản cũ thành *.before_engineering_fix.bak')
        print('Nhớ tải lên GitHub: app.py, uhpc_core.py, uhpc_engineering.py và uhpc_limits.py')
    except Exception:
        for path, backup in backs:
            if backup.exists():
                shutil.copy2(backup, path)
        raise


if __name__ == '__main__':
    main()
