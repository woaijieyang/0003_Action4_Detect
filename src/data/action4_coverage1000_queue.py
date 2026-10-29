"""Independent transactional queue; protects all existing dataset namespaces."""
import json,os,sqlite3,time
from data.action4_paths import PROJECT_ROOT as R

OUT=R/'outputs/coverage1000_v001'
DB=OUT/'slots.sqlite3'

def connect():
    db=sqlite3.connect(str(DB),timeout=30,isolation_level=None);db.row_factory=sqlite3.Row
    db.execute('PRAGMA busy_timeout=30000');return db

def initialize(plan):
    db=connect();db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS slots(slot INTEGER PRIMARY KEY,split TEXT,scene TEXT,spec TEXT,enabled INTEGER,status TEXT,attempt INTEGER,owner INTEGER,updated REAL,record TEXT,error TEXT)')
    expected={s['slot_id']:s for s in plan['slots']}
    if db.execute('SELECT count(*) FROM slots').fetchone()[0]:
        rows=db.execute('SELECT slot,spec FROM slots').fetchall()
        if len(rows)!=len(expected) or any(json.loads(r['spec'])!=expected[r['slot']] for r in rows):raise ValueError('Frozen queue identity differs')
    else:
        preview=set(plan['preview_slot_ids']);db.execute('BEGIN IMMEDIATE')
        for s in plan['slots']:
            start=int(plan.get('initial_attempt_by_slot',{}).get(str(s['slot_id']),0))
            db.execute('INSERT INTO slots VALUES(?,?,?,?,?,?,?,?,?,?,?)',(s['slot_id'],s['split'],s['scene_id'],json.dumps(s),int(s['slot_id'] in preview),'pending',start,None,time.time(),None,None))
        db.execute('COMMIT')
    db.close()

def claim(scene=None,quality=False):
    db=connect();db.execute('BEGIN IMMEDIATE')
    row=db.execute("SELECT * FROM slots WHERE status='rendered' ORDER BY updated LIMIT 1").fetchone() if quality else db.execute("SELECT * FROM slots WHERE enabled=1 AND status='pending' AND scene=? ORDER BY attempt,slot LIMIT 1",(scene,)).fetchone()
    if row:db.execute('UPDATE slots SET status=?,owner=?,updated=? WHERE slot=?',('checking' if quality else 'claimed',os.getpid(),time.time(),row['slot']))
    db.execute('COMMIT');db.close();return dict(row) if row else None

def next_attempt(slot):
    db=connect();db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT attempt,status,owner FROM slots WHERE slot=?',(slot,)).fetchone()
    if row['status']!='claimed' or row['owner']!=os.getpid():raise ValueError('Attempt without ownership')
    attempt=row['attempt'];db.execute('UPDATE slots SET attempt=?,updated=? WHERE slot=?',(attempt+1,time.time(),slot));db.execute('COMMIT');db.close();return attempt

def transition(slot,status,record=None,error=None):
    db=connect();db.execute('BEGIN IMMEDIATE');row=db.execute('SELECT owner FROM slots WHERE slot=?',(slot,)).fetchone()
    if row['owner']!=os.getpid():raise ValueError('Transition without ownership')
    db.execute('UPDATE slots SET status=?,record=?,error=?,owner=NULL,updated=? WHERE slot=?',(status,json.dumps(record) if record is not None else None,error,time.time(),slot));db.execute('COMMIT');db.close()

def pending_enabled(scene=None):
    db=connect();sql="SELECT count(*) FROM slots WHERE enabled=1 AND status IN ('pending','claimed','rendered','checking')";args=()
    if scene is not None:sql+=' AND scene=?';args=(scene,)
    n=db.execute(sql,args).fetchone()[0];db.close();return n

def counts():
    db=connect();r={x[0]:x[1] for x in db.execute('SELECT status,count(*) FROM slots GROUP BY status')};db.close();return r

def publish_manifests():
    cfg=json.loads((R/'configs/data/action4_coverage1000_v001.json').read_text());db=connect()
    for split in cfg['split_counts']:
        root=R/cfg['candidate_root']/split;root.mkdir(parents=True,exist_ok=True)
        rows=db.execute("SELECT record FROM slots WHERE status='accepted' AND split=? ORDER BY slot",(split,)).fetchall()
        dest=root/'annotations.jsonl';temp=dest.with_suffix('.jsonl.tmp')
        with temp.open('w') as f:
            for row in rows:f.write(row[0]+'\n')
            f.flush();os.fsync(f.fileno())
        os.replace(temp,dest)
    db.close()
