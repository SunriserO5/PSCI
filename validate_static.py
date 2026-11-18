"""Static checks only; never import project modules or execute notebook cells."""
from __future__ import annotations

import ast
import builtins
import hashlib
import json
import symtable
from pathlib import Path


def ast_digest(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def top_names(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(alias.asname or alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                names.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
    return names


def global_reads(table):
    result = {s.get_name() for s in table.get_symbols() if s.is_global() and s.is_referenced()}
    for child in table.get_children():
        result |= global_reads(child)
    return result


def audit():
    root = Path(__file__).resolve().parent
    expected = json.loads((root / 'copied_code_manifest.json').read_text(encoding='utf-8'))['definitions']
    baseline_path = root / 'workflow' / 'baseline.py'
    if baseline_path.exists():
        baseline = ast.parse(baseline_path.read_text(encoding='utf-8'))
        definitions = {n.name: n for n in baseline.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for name, info in expected.items():
            if name not in definitions or ast_digest(definitions[name]) != info['ast_sha256']:
                raise ValueError(f'复制实现与基线快照不同：{name}')
    notebooks = sorted(root.glob('*.ipynb'))
    if len(notebooks) != 7:
        raise ValueError('应有7本Notebook。')
    code_cells, copied_instances, verified = 0, 0, set()
    helper_definitions = {}
    for module in ('project', 'data'):
        helper = root / 'workflow' / f'{module}.py'
        if helper.exists():
            tree = ast.parse(helper.read_text(encoding='utf-8'))
            helper_definitions[module] = {n.name: ast_digest(n) for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    integration_copies_checked = 0
    switches = {'LOCK_ORIGINAL_MODELS', 'INSTALL_SPLITS', 'EVALUATE_DEVELOPMENT_SEGMENTATION',
                'SHOW_CACHE_EXAMPLE', 'COMPARE_DEVELOPMENT_OOF'}
    banned_roots = {'workflow', 'importlib', 'runpy', 'train_resnet50_manual_cv',
                    'train_resnet50_temporal_development', 'prep_resnet50_manual',
                    'prepare_temporal_validation', 'fusion_resnet50_manual_clinical5',
                    'train_resnet50_single_sequence_manual_cv', 'clinical5_model'}
    for path in notebooks:
        notebook = json.loads(path.read_text(encoding='utf-8'))
        if notebook['nbformat'] != 4 or notebook['nbformat_minor'] != 5:
            raise ValueError(f'格式不正确：{path.name}')
        ids, known, symbols = set(), set(dir(builtins)) | {'display', 'get_ipython'}, []
        notebook_definitions = {}
        for index, cell in enumerate(notebook['cells']):
            if cell['id'] in ids:
                raise ValueError(f'重复单元ID：{path.name}')
            ids.add(cell['id'])
            if cell['cell_type'] != 'code':
                continue
            code_cells += 1
            if cell['execution_count'] is not None or cell['outputs']:
                raise ValueError(f'存在执行状态或输出：{path.name}:{index}')
            source = ''.join(cell['source'])
            tree = ast.parse(source, filename=f'{path.name}:{index}')
            compile(source, f'{path.name}:{index}', 'exec', dont_inherit=True)
            table = symtable.symtable(source, f'{path.name}:{index}', 'exec')
            known.update(s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace())
            known.update(top_names(tree))
            symbols.append(table)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.level or (node.module or '').split('.')[0] in banned_roots:
                        raise ValueError(f'仍存在项目/动态导入：{path.name}:{index}')
                if isinstance(node, ast.Import):
                    if any(a.name.split('.')[0] in banned_roots for a in node.names):
                        raise ValueError(f'仍存在项目/动态导入：{path.name}:{index}')
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {'exec', 'eval', '__import__'}:
                    raise ValueError(f'不应动态执行源码：{path.name}:{index}')
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and (target.id.startswith('RUN_') or target.id in switches):
                            if not isinstance(node.value, ast.Constant) or node.value.value is not False:
                                raise ValueError(f'运行开关未关闭：{path.name}:{target.id}')
            definitions = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            for name, node in definitions.items():
                if name in notebook_definitions:
                    raise ValueError(f'函数或类被重复定义：{path.name}:{name}')
                notebook_definitions[name] = ast_digest(node)
            for name in cell.get('metadata', {}).get('copied_original_definitions', []):
                if name not in definitions or ast_digest(definitions[name]) != expected[name]['ast_sha256']:
                    raise ValueError(f'Notebook复制代码与基线快照不同：{path.name}:{name}')
                copied_instances += 1
                verified.add(name)
        missing = set().union(*(global_reads(t) for t in symbols)) - known
        if missing:
            raise ValueError(f'{path.name}存在未定义全局符号：{sorted(missing)}')
        for module in ('project', 'data') if path.name.startswith('00_') else ('project',):
            for name, checksum in helper_definitions.get(module, {}).items():
                if notebook_definitions.get(name) != checksum:
                    raise ValueError(f'Notebook与维护副本不同：{path.name}:{module}.{name}')
                integration_copies_checked += 1
    if verified != set(expected):
        raise ValueError(f'核心实现未全部复制到Notebook：{sorted(set(expected)-verified)}')
    config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
    sample_path = root / 'config.sample.json'
    if sample_path.exists():
        sample = json.loads(sample_path.read_text(encoding='utf-8'))
        if sample['preprocessing']['mode'] != 'sample' or sample['run_id'] == config['run_id']:
            raise ValueError('样例应使用独立实验目录及sample模式。')
    # Auxiliary maintenance files are parsed only, not imported or required by notebooks.
    helpers = list((root / 'workflow').glob('*.py')) if (root / 'workflow').exists() else []
    for helper in helpers:
        ast.parse(helper.read_text(encoding='utf-8'), filename=str(helper))
    return {'notebooks': len(notebooks), 'code_cells': code_cells,
            'unique_original_definitions_copied': len(verified), 'copied_instances': copied_instances,
            'integration_definition_copies_checked': integration_copies_checked,
            'project_imports': 0, 'undefined_notebook_globals': 0,
            'original_code_directory_required': False, 'workflow_package_required': False,
            'checks': 'syntax, symbol tables, original/integration AST copies, JSON, empty outputs, disabled switches',
            'execution': 'not performed'}


if __name__ == '__main__':
    print(json.dumps(audit(), ensure_ascii=False, indent=2))
