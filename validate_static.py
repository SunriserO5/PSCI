"""Static checks only; never import project modules or execute notebook cells."""
from __future__ import annotations

import ast
import argparse
import builtins
import hashlib
import json
import symtable
from pathlib import Path


def ast_digest(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def helper_digest(node):
    class WithoutProjectImports(ast.NodeTransformer):
        def visit_ImportFrom(self, item):
            return None if item.level else item
    import copy
    return ast_digest(WithoutProjectImports().visit(copy.deepcopy(node)))


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


def audit(notebook03_only=False, allow_existing_outputs=False):
    root = Path(__file__).resolve().parent
    expected = json.loads((root / 'copied_code_manifest.json').read_text(encoding='utf-8'))['definitions']
    baseline_path = root / 'workflow' / 'baseline.py'
    if baseline_path.exists():
        baseline = ast.parse(baseline_path.read_text(encoding='utf-8'))
        baseline_definitions = {n.name: n for n in baseline.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        for name, info in expected.items():
            if name not in baseline_definitions or ast_digest(baseline_definitions[name]) != info['ast_sha256']:
                raise ValueError(f'复制实现与基线快照不同：{name}')
    notebooks = sorted(root.glob('*.ipynb'))
    if len(notebooks) != 7:
        raise ValueError('应有7本Notebook。')
    if notebook03_only:
        notebooks = [p for p in notebooks if p.name.startswith('03_')]
    code_cells, copied_instances, verified = 0, 0, set()
    helper_definitions = {}
    for module in ('project', 'data', 'ablations', 'selected_slices', 'radiomics', 'segmentation', 'optimized', 'original', 'explain', 'reporting'):
        helper = root / 'workflow' / f'{module}.py'
        if helper.exists():
            tree = ast.parse(helper.read_text(encoding='utf-8'))
            helper_definitions[module] = {n.name: helper_digest(n) for n in tree.body
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
            if not allow_existing_outputs and (cell['execution_count'] is not None or cell['outputs']):
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
                    if node.level or ((node.module or '').split('.')[0] in banned_roots and node.module != 'importlib.metadata'):
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
        modules = ('project', 'data') if path.name.startswith('00_') else ('project',)
        if path.name.startswith('03_') and 'ImageOnlyAttentionModel' in notebook_definitions:
            modules += ('ablations',)
        if path.name.startswith('01_'):
            modules += ('selected_slices', 'radiomics')
        if path.name.startswith('02_'):
            modules += ('optimized',)
        if path.name.startswith('05_'):
            modules += ('selected_slices', 'segmentation')
        if path.name.startswith('06_'):
            modules += ('selected_slices', 'segmentation', 'reporting')
        # Every copied helper definition must match its maintenance source wherever present.
        for module, definitions in helper_definitions.items():
            for name, checksum in definitions.items():
                if name in notebook_definitions and notebook_definitions[name] != checksum:
                    raise ValueError(f'辅助定义副本不一致：{path.name}:{module}.{name}')
        for module in modules:
            for name, checksum in helper_definitions.get(module, {}).items():
                if notebook_definitions.get(name) != checksum:
                    raise ValueError(f'Notebook与维护副本不同：{path.name}:{module}.{name}')
                integration_copies_checked += 1
    if not notebook03_only and verified != set(expected):
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
    if (root / 'workflow' / 'ablations.py').is_file() and baseline_path.is_file():
        ablation_tree = ast.parse((root / 'workflow' / 'ablations.py').read_text(encoding='utf-8'))
        ablations = {n.name: n for n in ablation_tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        original_body = ast.Module(body=baseline_definitions['train_development_fold'].body[1:], type_ignores=[])
        new_body = ast.Module(body=ablations['train_structure_fold'].body[2:], type_ignores=[])
        if ast_digest(original_body) != ast_digest(new_body):
            raise ValueError('结构消融的训练循环偏离原开发训练循环。')
        no_mask = next(n for n in ablations['ImageOnlyAttentionModel'].body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        if any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == 'masks' for n in ast.walk(no_mask)):
            raise ValueError('去病灶组仍在读取掩模内容。')
        mean_pool = next(n for n in ablations['MeanPoolingDualBranchModel'].body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
        if any(isinstance(n, ast.Attribute) and n.attr == 'attention' for n in ast.walk(mean_pool)):
            raise ValueError('去注意力组仍在调用注意力模块。')
        if not any(isinstance(n, ast.Call) and ast.unparse(n) == 'feat.mean(dim=1)' for n in ast.walk(mean_pool)):
            raise ValueError('去注意力组应沿切片维度取平均。')
    return {'notebooks': len(notebooks), 'code_cells': code_cells,
            'unique_original_definitions_copied': len(verified), 'copied_instances': copied_instances,
            'integration_definition_copies_checked': integration_copies_checked,
            'project_imports': 0, 'undefined_notebook_globals': 0,
            'original_code_directory_required': False, 'workflow_package_required': False,
            'scope': 'notebook03_update' if notebook03_only else 'all_notebooks',
            'structural_training_loop_matches_baseline': (root / 'workflow' / 'ablations.py').is_file(),
            'checks': 'syntax, symbol tables, original/integration AST copies, JSON, preserved outputs if requested, disabled switches',
            'execution': 'not performed'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--notebook03-update', action='store_true', help='只核查03更新，保留其他Notebook的用户运行状态。')
    parser.add_argument('--allow-existing-outputs', action='store_true', help='保留用户已有执行结果，只做源码静态核查。')
    args = parser.parse_args()
    print(json.dumps(audit(notebook03_only=args.notebook03_update, allow_existing_outputs=args.allow_existing_outputs), ensure_ascii=False, indent=2))
