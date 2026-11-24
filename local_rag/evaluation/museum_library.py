"""Build a private pilot from the user's four explicitly secondary compilations.

Only the repeated featured-work layout is parsed. Editorial NO. values are never
accession numbers, and research suggestions are not promoted to established facts.
"""

import json
from pathlib import Path
import re
from uuid import uuid4

from local_rag.core.config import RUNTIME_DIR
from local_rag.documents.reader import DocumentReader
from local_rag.evaluation.museum_demo import digest, write_json, load_dataset

COLLECTIONS = {
    'museum_nanjing_85.pdf': '南京博物院',
    'museum_palace_58.pdf': '故宫博物院',
    'museum_shanghai_41.pdf': '上海博物馆',
    'museum_national_132.pdf': '中国国家博物馆',
}
ENTRY = re.compile(r'(?m)^NO\. (\d{3}) 馆方资料概述\n([^\n]+)\n([^\n]+)\n')


def featured_claims(reading, filename, museum):
    """Extract literal fields from the page that explicitly names this museum."""
    if '不是馆方出版的原版图录' not in reading.text or '非馆方原版出版物' not in reading.text:
        raise ValueError(f'Unrecognized source classification in {filename}.')
    name = 'corpus/' + filename
    records = []
    for unit in reading.extraction.units:
        matches = list(ENTRY.finditer(unit.text))
        if not matches:
            continue
        if not unit.text.startswith(museum + ' 馆藏文物资料汇编 / 重点文物'):
            raise ValueError(f'Featured page museum mismatch in {filename}, page {unit.page_number}.')
        for n, match in enumerate(matches):
            stop = matches[n + 1].start() if n + 1 < len(matches) else unit.text.index('学习型资料 · 非馆方原版出版物', match.end())
            block = unit.text[match.start():stop].strip()
            era, marker, medium = match[3].partition(' · 精品详解 · ')
            if not marker or not era or not medium:
                raise ValueError(f'Unrecognized date/material layout: {match[2]}')
            # Preserve the size statement and narrative together; observation prompts remain in raw PDF.
            description = block.split('公开尺寸/形制记载：', 1)[1].split('\n阅读观察 ', 1)[0]
            description = '公开尺寸/形制记载：' + description
            fields = [('name', match[2]), ('era', era), ('material', medium), ('description', description)]
            claims = [{'file': name, 'field': field, 'value': value, 'quote': block}
                      for field, value in fields if not re.search(r'待核对|未公开|不详', value) or field == 'description']
            # The page header establishes collection context, not the filename or excavation site.
            claims.append({'file': name, 'field': 'museum', 'value': museum,
                           'quote': unit.text[:stop].strip()})
            records.append({'key': filename.removesuffix('.pdf') + '_' + match[1],
                            'source_page': unit.page_number, 'editorial_sequence': match[1], 'claims': claims})
    if len(records) != 8 or {r['editorial_sequence'] for r in records} != {f'{i:03}' for i in range(1, 9)}:
        raise ValueError(f'Expected eight featured works in {filename}; refusing to guess layout.')
    return records


def questions():
    files = dict(n='museum_nanjing_85.pdf', p='museum_palace_58.pdf', s='museum_shanghai_41.pdf', g='museum_national_132.pdf')
    cases = []
    def turn(q, expected, facts=(), sources=(), reset=False):
        return {'question': q, 'reference_answer': expected,
                'fact_groups': [[x] if isinstance(x, str) else x for x in facts],
                'source_files': ['corpus/' + files[x] for x in sources], 'reset_history': reset}
    def case(id, audience, tags, *turns):
        cases.append({'id': id, 'audience': audience, 'tags': tags,
                      'critical': bool(set(tags) & {'ambiguity', 'conflict', 'missing_evidence', 'isolation'}),
                      'turns': list(turns)})
    case('v01','visitor',['explanation'],turn('请依据资料，用简单的话介绍西汉金兽的材料和外形。','汇编记金，呈团伏兽形，器表有锤饰斑纹；不添加功能猜测。',['金','团伏'],['n']))
    case('v02','visitor',['lookup'],turn('莲鹤方壶是什么年代和材料？','汇编记春秋、青铜。',['春秋','青铜'],['p']))
    case('v03','visitor',['explanation'],turn('鹰形陶鼎靠什么站稳？请用简单的话解释。','尾部和双腿构成稳定支点。',['尾部',['双腿','两腿']],['g']))
    case('v04','visitor',['lookup'],turn('舞蹈纹彩陶盆画了几组人物，每组几人？','三组，每组五人。',[['三组','3组'],['五人','5人']],['g']))
    case('v05','visitor',['purpose'],turn('子仲姜盘是做什么用的？','据汇编，为子仲姜制作的盥洗器。',['盥洗'],['s']))
    case('v06','visitor',['purpose'],turn('上海博物馆的牺尊可以用来做什么？','据汇编可用于温酒；背上容器及注水空间与此有关。',['温酒'],['s']))
    case('v07','visitor',['followup'],turn('金蝉玉叶饰件用了哪些材料？','金、和田玉。',['金','和田玉'],['n']),turn('它收藏在哪个博物馆？','该汇编将它列在南京博物院馆藏中，不能将出土地苏州当收藏馆。',['南京博物院'],['n']))
    case('v08','visitor',['explanation'],turn('大克鼎的铭文记载了什么？','贵族克接受册命及赏赐。',['册命','赏赐'],['s']))
    case('v09','visitor',['lookup'],turn('击鼓说唱俑是什么年代、什么材料？','东汉、陶。',['东汉','陶'],['g']))
    case('v10','visitor',['isolation'],turn('莲鹤方壶是什么材料？','青铜。',['青铜'],['p']),turn('它是什么材料？','新对话没有对象，应澄清文物名称或明确资料不足，不能继承上一对话。',[],[],True))
    case('s01','staff',['location','followup'],turn('西汉金兽的出土地和收藏单位分别是什么？','江苏盱眙地区窖藏；本汇编收藏单位南京博物院，二者应区分。',['盱眙','南京博物院'],['n']),turn('它的重量在资料里记载为多少？','9100克。',['9100'],['n']))
    case('s02','staff',['missing_evidence'],turn('南京博物院汇编中西汉金兽前的 NO. 001 是正式馆藏编号吗？它的正式馆藏编号是什么？','NO. 001 是本册条目顺序，不是已核验馆藏编号；资料未提供正式馆藏编号。',[],['n']))
    case('s03','staff',['missing_evidence'],turn('东汉错银铜牛灯的尺寸在这份资料里有记载吗？','条目写馆方首页资料未列尺寸，不能补造尺寸。',[],['n']))
    case('s04','staff',['lookup'],turn('明洪武釉里红寿石花卉纹大瓷盘的口径和高度是多少？','口径58.3厘米，高10.1厘米。',['58.3','10.1'],['n']))
    case('s05','staff',['lookup'],turn('中国国家博物馆的青铜冰鉴长宽和高度是多少？','长宽均76厘米，高63.2厘米。',['76','63.2'],['g']))
    case('s06','staff',['lookup'],turn('故宫博物院的霁红釉梅瓶底部有什么款识？','大清康熙年制。',['大清康熙年制'],['p']))
    case('s07','staff',['lookup'],turn('商鞅方升在资料中的年代和容积是什么？','战国、秦孝公十八年（前344）；约202毫升，保留约数限定。',['前344','202',['约','大约']],['s']))
    case('s08','staff',['missing_evidence'],turn('上传的四份博物馆资料是馆方出版的原版图录吗？','均明确为编者学习型汇编，不是馆方原版图录；不能因标题或链接是官方就升级为正式档案。',[['汇编','编者'],['不是','并非','非馆方']],['n','p','s','g']))
    case('s09','staff',['lookup'],turn('东汉玉严印有几面刻字，合计多少字？','四面，合计三十二字。',[['四面','4面'],['三十二','32']],['n']))
    case('s10','staff',['topic_change'],turn('莲鹤方壶是什么材料？','青铜。',['青铜'],['p']),turn('换个文物，三彩釉陶载乐骆驼是什么年代？','唐、开元十一年（723）。',['唐','723'],['g']),turn('它收藏在哪里？','此时指三彩釉陶载乐骆驼，依据汇编为中国国家博物馆。',['中国国家博物馆'],['g']))
    case('r01','researcher',['comparison'],turn('比较金蝉玉叶饰件与镶嵌兽面纹牌饰的复合材料和制作方式。','金蝉玉叶：黄金蝉栖薄玉叶，金工加琢玉；牌饰：青铜框架承托绿松石拼嵌。',['金','玉','青铜','绿松石'],['n','s']))
    case('r02','researcher',['lookup'],turn('“后母戊”青铜方鼎的铭文和制造技术在资料中如何描述？','内壁铸后母戊，涉及大型陶范浇铸与组织分工。',['后母戊','陶范','分工'],['g']))
    case('r03','researcher',['missing_evidence'],turn('这些汇编中的扩展导读能作为逐件核验过的鉴定报告吗？','不能；阅读说明明确扩展导读不等同逐件核验鉴定报告，来源可能只是分类或目录页。',[['不能','不等同'],['目录','分类']],['n','p','s','g']))
    case('r04','researcher',['comparison'],turn('依据故宫汇编，比较《五牛图》与《清明上河图》的画面主题。','五牛图以五头不同姿态牛为主体；清明上河图描述北宋汴京城郊、汴河和街市的城市活动。',[['五头','五牛'],'汴京','汴河'],['p']))
    case('r05','researcher',['missing_evidence'],turn('舞蹈纹彩陶盆的正式馆藏编号在上传资料中是什么？','上传资料未给正式馆藏编号，不能用 NO. 002 或系统 WW 编号替代。',[],['g']))
    case('r06','researcher',['missing_evidence'],turn('仅根据上传资料，西汉金兽今天在第几号展厅展出？','资料没有实时展厅信息，不能由收藏馆推断今天展厅。',[],['n']))
    case('r07','researcher',['followup'],turn('青铜冰鉴的内外层怎样组合？','外层方鉴、内层方尊缶，通过榫眼与钩扣固定。',['方鉴','方尊缶','榫眼','钩扣'],['g']),turn('这种结构有什么用途？','夹层置冰、内器盛酒；不要把其他青铜器温酒结构附给它。',['冰','酒'],['g']))
    case('r08','researcher',['comparison'],turn('鹰形陶鼎和红山文化玉龙分别属于什么文化、采用什么材料？','鹰形陶鼎：仰韶文化、陶；红山文化玉龙：红山文化、玉。',['仰韶','红山','陶','玉'],['g']))
    case('r09','researcher',['isolation'],turn('大克鼎收藏在哪里？','上海博物馆，按汇编来源表达。',['上海博物馆'],['s']),turn('它的铭文说了什么？','新对话没有指代对象，应追问名称或表示资料不足，不沿用大克鼎。',[],[],True))
    case('r10','researcher',['ambiguity'],turn('这件青铜器的用途是什么？','无历史、未指定具体青铜器，多件可能，应澄清名称而非任意选择某件。',[],[]))
    return cases


def build_dataset(source_dir):
    """Snapshot only discovered original PDFs; leave the user's library untouched."""
    source_dir = Path(source_dir).resolve()
    discovered = {}
    for metadata in source_dir.glob('*/document.json'):
        row = json.loads(metadata.read_text(encoding='utf-8'))
        filename = row.get('filename')
        if filename not in COLLECTIONS:
            continue
        if filename in discovered:
            raise ValueError(f'Multiple library documents named {filename}; select an unambiguous source directory.')
        original = metadata.parent / 'original.pdf'
        if original.resolve().parent != metadata.parent.resolve():
            raise ValueError('Original PDF must belong to its document directory.')
        raw = original.read_bytes()
        if digest(raw) != row['sha256']:
            raise ValueError(f'Original PDF hash differs from its library metadata: {filename}')
        reading = DocumentReader().read(original, filename=filename)
        records = featured_claims(reading, filename, COLLECTIONS[filename])
        discovered[filename] = (row, raw, reading, records, original)
    if set(discovered) != set(COLLECTIONS):
        raise ValueError('Missing museum compilations: ' + ', '.join(sorted(set(COLLECTIONS) - set(discovered))))
    root = RUNTIME_DIR / 'datasets' / ('museum-documents-' + uuid4().hex)
    (root / 'corpus').mkdir(parents=True)
    sources, records, inventory = [], [], []
    for filename, (row, raw, reading, featured, original) in discovered.items():
        target = root / 'corpus' / filename
        target.write_bytes(raw)
        sources.append({'file': 'corpus/' + filename, 'source_type': 'secondary', 'sha256': digest(raw),
                        'original_document_id': row['document_id']})
        records.extend(featured)
        inventory.append({'filename': filename, 'original_path': str(original), 'original_document_id': row['document_id'],
                          'sha256': digest(raw), 'pages': reading.extraction.page_count, 'featured_records': len(featured),
                          'museum': COLLECTIONS[filename], 'source_type': 'secondary'})
    data = {'version': 1, 'benchmark_id': 'museum_documents_v1', 'synthetic': False,
            'disclaimer': '依据用户上传的学习型二次汇编；未经馆方原始档案逐条核验，NO. 序号不是馆藏编号。',
            'museums': list(COLLECTIONS.values()), 'sources': sources, 'artifacts': records, 'cases': questions()}
    path = root / 'dataset.json'
    write_json(path, data)
    write_json(root / 'source_inventory.json', inventory)
    load_dataset(path)
    return path
