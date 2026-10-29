"""Transactional slot/attempt ownership for persistent fixed-scene workers."""
import json,sqlite3,time,os
from pathlib import Path
from data.action4_paths import PROJECT_ROOT as R,contained_path

DB=R/"outputs/phrase2_generation_v001/slots.sqlite3"
def connect():
    db=sqlite3.connect(str(DB),timeout=30,isolation_level=None);db.row_factory=sqlite3.Row;db.execute("PRAGMA busy_timeout=30000");return db
def initialize(plan):
    DB.parent.mkdir(exist_ok=True)
    db=connect();db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE IF NOT EXISTS slots(slot INTEGER PRIMARY KEY,split TEXT,scene TEXT,spec TEXT,enabled INTEGER,status TEXT,attempt INTEGER,owner INTEGER,updated REAL,record TEXT,error TEXT)")
    if db.execute("SELECT count(*) FROM slots").fetchone()[0]:
        rows=db.execute("SELECT slot,spec FROM slots").fetchall();expected={r["slot_id"]:r for r in plan["slots"]}
        if len(rows)!=len(expected) or any(json.loads(r["spec"])!=expected[r["slot"]] for r in rows):raise ValueError("Existing queue conflicts with fixed plan")
    else:
        preview=set(plan["preview_slot_ids"]);db.execute("BEGIN IMMEDIATE")
        for r in plan["slots"]:db.execute("INSERT INTO slots VALUES(?,?,?,?,?,?,?,?,?,?,?)",(r["slot_id"],r["split"],r["scene_id"],json.dumps(r),int(r["slot_id"] in preview),"pending",0,None,time.time(),None,None))
        db.execute("COMMIT")
    db.close()
def claim(scene=None,quality=False):
    db=connect();db.execute("BEGIN IMMEDIATE")
    if quality:row=db.execute("SELECT * FROM slots WHERE status='rendered' ORDER BY updated LIMIT 1").fetchone()
    else:row=db.execute("SELECT * FROM slots WHERE enabled=1 AND status='pending' AND scene=? ORDER BY attempt,slot LIMIT 1",(scene,)).fetchone()
    if row:
        status="checking" if quality else "claimed"
        db.execute("UPDATE slots SET status=?,owner=?,updated=? WHERE slot=?",(status,os.getpid(),time.time(),row["slot"]))
    db.execute("COMMIT");db.close();return dict(row) if row else None
def next_attempt(slot):
    db=connect();db.execute("BEGIN IMMEDIATE");row=db.execute("SELECT attempt,status,owner FROM slots WHERE slot=?",(slot,)).fetchone()
    if row["status"]!="claimed" or row["owner"]!=os.getpid():raise ValueError("Worker does not own slot")
    attempt=row["attempt"];db.execute("UPDATE slots SET attempt=?,updated=? WHERE slot=?",(attempt+1,time.time(),slot));db.execute("COMMIT");db.close();return attempt
def transition(slot,status,record=None,error=None):
    db=connect()
    row=db.execute("SELECT owner FROM slots WHERE slot=?",(slot,)).fetchone()
    if row["owner"]!=os.getpid():raise ValueError("Slot transition from non-owner")
    db.execute("UPDATE slots SET status=?,record=?,error=?,owner=NULL,updated=? WHERE slot=?",(status,json.dumps(record) if record is not None else None,error,time.time(),slot));db.close()
def counts():
    db=connect();rows={r[0]:r[1] for r in db.execute("SELECT status,count(*) FROM slots GROUP BY status")};db.close();return rows
def open_bulk():
    db=connect();db.execute("UPDATE slots SET enabled=1");db.close()
def publish_manifests():
    db=connect()
    for split in ["train","eval","test"]:
        directory=R/"data/phrase2"/split;directory.mkdir(parents=True,exist_ok=True)
        rows=db.execute("SELECT record FROM slots WHERE status='accepted' AND split=? ORDER BY slot",(split,)).fetchall()
        target=directory/"annotations.jsonl";temp=target.with_name("annotations.jsonl.tmp")
        with temp.open("w") as f:
            for row in rows:f.write(row[0]+"\n")
            f.flush();os.fsync(f.fileno())
        os.replace(temp,target)
    db.close()
