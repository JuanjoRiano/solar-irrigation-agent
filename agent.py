import argparse, csv, json, math, statistics, time
from pathlib import Path
from datetime import datetime, timedelta
from collections import Counter, deque
from dataclasses import dataclass, replace
from itertools import product

BASE = Path(__file__).resolve().parent
POLICY = {'threshold': 0.60, 'cost_solar': 8.0, 'cost_backup': 28.0,
          'cost_reinforce': 12.0, 'benefit': 100.0, 'deficit_penalty': 40.0}
ACTIONS = ('solar', 'respaldo', 'posponer')
LABELS = {'solar':'Programar solar', 'respaldo':'Programar con respaldo', 'posponer':'Posponer y monitorear'}


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def write_csv(path, rows):
    if not rows: return
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def load_data():
    raw = json.loads((BASE/'data/nasa_power_raw.json').read_text())
    src=raw['properties']['parameter']; rows=[]; missing=[]
    for key in sorted(src['ALLSKY_SFC_SW_DWN']):
        vals=[src[k].get(key, -999) for k in ('ALLSKY_SFC_SW_DWN','PRECTOTCORR','T2M')]
        if any(x == raw['header']['fill_value'] or not math.isfinite(x) for x in vals):
            missing.append(key); continue
        rows.append({'date':datetime.strptime(key,'%Y%m%d').strftime('%Y-%m-%d'),
                     'solar_kwh_m2':vals[0], 'rain_mm':vals[1], 'temp_c':vals[2]})
    train_days=[r for r in rows if r['date']<'2025-01-01']
    thresholds={'solar':statistics.median(r['solar_kwh_m2'] for r in train_days),
                'rain':statistics.median(r['rain_mm'] for r in train_days),
                'rain_q75':statistics.quantiles([r['rain_mm'] for r in train_days],n=4)[2]}
    pairs=[]
    for a,b in zip(rows,rows[1:]):
        if datetime.fromisoformat(b['date'])-datetime.fromisoformat(a['date']) != timedelta(days=1):continue
        pairs.append({**a,'next_date':b['date'],'C':int(a['rain_mm']<=thresholds['rain']),
                      'E':int(a['solar_kwh_m2']>=thresholds['solar']),
                      'R':int(b['rain_mm']<=thresholds['rain'] and b['solar_kwh_m2']>=thresholds['solar'])})
    train=[r for r in pairs if r['next_date']<'2025-01-01']
    test=[r for r in pairs if r['date']>='2025-01-01']
    return rows,train,test,thresholds,missing,raw


def fit_bayes(train):
    #Topología C->E, C->R, E->R. Frecuencias + Laplace alpha=1.
    n=len(train); nc=Counter(x['C'] for x in train)
    nce=Counter((x['C'],x['E']) for x in train)
    ncer=Counter((x['C'],x['E'],x['R']) for x in train)
    joint={}; audit=[]
    for c,e,r in product((0,1),repeat=3):
        pc=(nc[c]+1)/(n+2); pe=(nce[c,e]+1)/(nc[c]+2)
        pr=(ncer[c,e,r]+1)/(nce[c,e]+2)
        joint[c,e,r]=pc*pe*pr
        audit.append({'C':c,'E':e,'R':r,'n_C':nc[c],'n_CE':nce[c,e],
                      'n_CER':ncer[c,e,r],'P_C':pc,'P_E_given_C':pe,'P_R_given_CE':pr,'joint':joint[c,e,r]})
    return joint,audit


def posterior(joint,c,e,prior=None):
    if prior is not None and not 0<=prior<=1:raise ValueError('Prior fuera de [0,1].')
    masses={r:sum(p for (cc,ee,rr),p in joint.items() if rr==r) for r in (0,1)}
    def weight(r):
        base=joint[c,e,r]
        return base if prior is None else base/masses[r]*(prior if r else 1-prior)
    den=weight(0)+weight(1)
    if den==0:raise ValueError('Evidencia con probabilidad cero.')
    return weight(1)/den


@dataclass(frozen=True)
class Op:
    name:str; pre:frozenset; add:frozenset; delete:frozenset=frozenset()

def op(name,pre,add,delete=()):return Op(name,frozenset(pre),frozenset(add),frozenset(delete))
OPS=(op('Ir a parcela',{'base'},{'parcela'},{'base'}),
     op('Verificar equipo',{'parcela','kit'},{'verificado'}),
     op('Revisar sensores',{'parcela','kit'},{'sensores_revisados'}),
     op('Preparar respaldo',{'verificado'},{'respaldo_listo'}),
     op('Programar solar',{'verificado'},{'solar_programado'},{'pendiente'}),
     op('Programar respaldo',{'verificado','respaldo_listo'},{'respaldo_programado'},{'pendiente'}),
     op('Registrar espera',{'parcela'},{'espera_registrada'},{'pendiente'}),
     op('Inspeccionar paneles',{'base'},{'paneles_inspeccionados'}))
INITIAL=frozenset({'base','kit','pendiente'})


def strips(p,action,threshold=None):
    #BFS real: prepara cada alternativa antes de que Minimax seleccione.
    threshold=POLICY['threshold'] if threshold is None else threshold
    target={'solar':'solar_programado','respaldo':'respaldo_programado','posponer':'espera_registrada'}[action]
    goal={target}
    if p<threshold:goal.add('sensores_revisados')
    #Incluir solo el operador terminal de esta alternativa evita planes contradictorios.
    allowed=[a for a in OPS if not (a.add & {'solar_programado','respaldo_programado','espera_registrada'}) or target in a.add]
    q=deque([(INITIAL,[],[INITIAL])]); seen={INITIAL}
    while q:
        state,plan,states=q.popleft()
        if goal<=state:return {'goal':sorted(goal),'plan':plan,'states':[sorted(x) for x in states]}
        for a in allowed:
            if a.pre<=state:
                new=(state-a.delete)|a.add
                if new not in seen:seen.add(new);q.append((new,plan+[a.name],states+[new]))
    raise ValueError('No existe plan STRIPS.')


def validate_plan(result):
    state=INITIAL
    for name in result['plan']:
        a=next(x for x in OPS if x.name==name)
        assert a.pre<=state
        state=(state-a.delete)|a.add
    assert set(result['goal'])<=state and sorted(state)==result['states'][-1]


@dataclass(frozen=True)
class State:
    action:str=''; weather1:int=-1; reinforce:bool=False; weather2:int=-1; depth:int=0


def representative_days(train):
    ordered=sorted(train,key=lambda r:(r['solar_kwh_m2'],r['date']))
    #Registros completos reales: no se mezclan radiación y lluvia de fechas distintas.
    return [ordered[round((len(ordered)-1)*q)] for q in (.1,.5,.9)]


def children(s):
    if s.depth==0:return [(LABELS[a],replace(s,action=a,depth=1)) for a in ACTIONS]
    if s.depth==1:return [(f"Ambiente {i+1}",replace(s,weather1=i,depth=2)) for i in range(3)]
    if s.depth==2:return [('Mantener',replace(s,depth=3)),('Reforzar',replace(s,reinforce=True,depth=3))]
    if s.depth==3:return [(f'Ventana 2 / ambiente {i+1}',replace(s,weather2=i,depth=4)) for i in (0,2)]
    return []


def utility(s,p,days,thresholds,heuristic=False):
    #Puntos de política (no pesos/costos monetarios medidos).
    #Radiación y lluvia vienen de registros reales. Cobertura y utilidad son indicadores de diseño; NO son caudal ni energía eléctrica observados.
    
    if not s.action:return 0.
    ids=[s.weather1 if s.weather1>=0 else 1,s.weather2 if s.weather2>=0 else 1]
    #Heurística del horizonte no observado: usar escenario central.
    benefits=[]
    for idx in ids:
        day=days[idx]
        solar=min(1.,day['solar_kwh_m2']/thresholds['solar'])
        need=max(0.,1-day['rain_mm']/max(thresholds['rain_q75'],.001))
        weight=.5+.5*need
        if s.action=='posponer':coverage=0.
        elif s.action=='solar':coverage=solar
        else:coverage=min(1.,solar+.35)
        if s.reinforce and s.action!='posponer':coverage=min(1.,coverage+.20)
        benefits.append(weight*(POLICY['benefit']*coverage-POLICY['deficit_penalty']*(1-coverage)))
    cost={'solar':POLICY['cost_solar'],'respaldo':POLICY['cost_backup'],'posponer':0}[s.action]
    if s.reinforce:cost+=POLICY['cost_reinforce']
    return p*statistics.mean(benefits)-cost


def minimax(p,days,t,depth=4,variant='naive'):
    #Naive/alpha-beta comparan el mismo horizonte. Heurística corta una capa antes.
    if depth not in (2,3,4):raise ValueError('Profundidad debe ser 2, 3 o 4.')
    if variant not in ('naive','alpha_beta','heuristic'):raise ValueError('Variante desconocida.')
    limit=depth-1 if variant=='heuristic' else depth
    counts={'visited':0,'expanded':0,'evaluated':0,'pruned_edges':0}
    def walk(s,alpha=-math.inf,beta=math.inf):
        counts['visited']+=1
        if s.depth==limit:
            counts['evaluated']+=1
            return utility(s,p,days,t,variant=='heuristic'),[]
        counts['expanded']+=1
        ismax=s.depth%2==0; best=-math.inf if ismax else math.inf; path=[]
        kids=children(s)
        for i,(label,ch) in enumerate(kids):
            val,tail=walk(ch,alpha,beta)
            if (ismax and val>best) or (not ismax and val<best):best=val;path=[label]+tail
            if variant=='alpha_beta':
                if ismax:alpha=max(alpha,best)
                else:beta=min(beta,best)
                if beta<=alpha:
                    counts['pruned_edges']+=len(kids)-i-1;break
        return best,path
    value,path=walk(State())
    action=next(a for a in ACTIONS if LABELS[a]==path[0])
    return {'decision':action,'value':value,'path':path,'depth_requested':depth,'depth_effective':limit,**counts}


def exact_action_value(action,p,days,t):
    return min(max(min(utility(State(action,i,reinforce,j,4),p,days,t) for j in (0,2)) for reinforce in (False,True)) for i in range(3))


def flow(record,joint,days,t,prior=None):
    p=posterior(joint,record['C'],record['E'],prior)
    candidates={a:strips(p,a) for a in ACTIONS}
    for plan in candidates.values():validate_plan(plan)
    decision=minimax(p,days,t,4,'alpha_beta')
    return {'date':record['date'],'evidence':{'C':record['C'],'E':record['E']},'posterior':p,
            'candidates':candidates,'minimax':decision,'selected_plan':candidates[decision['decision']]}


def benchmark(cases,joint,days,t,repeats=31):
    rows=[]
    for case in cases:
        p=posterior(joint,case['C'],case['E']);full=minimax(p,days,t,4)
        for depth in (2,3,4):
            base=minimax(p,days,t,depth)
            for variant in ('naive','alpha_beta','heuristic'):
                minimax(p,days,t,depth,variant);times=[]
                for _ in range(repeats):
                    start=time.perf_counter_ns();r=minimax(p,days,t,depth,variant)
                    times.append((time.perf_counter_ns()-start)/1e6)
                realized=exact_action_value(r['decision'],p,days,t)
                rows.append({'scenario':case['date'],'C':case['C'],'E':case['E'],'posterior':p,
                             'variant':variant,'depth':depth,'effective_depth':r['depth_effective'],
                             'median_ms':statistics.median(times),'expanded':r['expanded'],
                             'visited':r['visited'],'evaluated':r['evaluated'],
                             'decision':r['decision'],'same_as_naive':r['decision']==base['decision'],
                             'value':r['value'],'full_value_action':realized,'regret_vs_full':full['value']-realized})
    return rows


def charts(out,result,joint,days,t,sensitivity):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    green='#245D48';light='#CBDDD3';red='#A64A3C';p=result['posterior']
    # Bayes: probabilities, explicit evidence, units and target.
    fig,ax=plt.subplots(figsize=(9,5))
    vals=[p,1-p];ax.bar(['Ventana meteorológica apta','Ventana no apta'],vals,color=[green,red]);ax.set_ylim(0,1.13)
    for i,v in enumerate(vals):ax.text(i,v+.025,f'{v:.2%}',ha='center',fontweight='bold',fontsize=17)
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_ylabel('Probabilidad posterior')
    ax.set_title(f"Bayes | {result['date']} | C={result['evidence']['C']}, E={result['evidence']['E']}")
    fig.text(.5,.015,'Apta = lluvia mañana ≤ mediana y radiación mañana ≥ mediana de entrenamiento. No mide éxito real del riego.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(out/'bayes_bars.png',dpi=180);plt.close(fig)
    #Full Minimax tree: 67 nodes, all values, selected principal variation.
    records=[];edges=[];counter=0
    def build(s,label,parent=None):
        nonlocal counter
        idx=counter;counter+=1
        node={'id':idx,'s':s,'label':label};records.append(node)
        if parent is not None:edges.append((parent,idx,label))
        ch=[]
        for lab,child in children(s):ch.append(build(child,lab,idx))
        node['children']=ch
        node['value']=utility(s,p,days,t) if not ch else (max if s.depth%2==0 else min)(records[j]['value'] for j in ch)
        return idx
    build(State(),'MAX')
    xpos={};nleaf=0
    def place(i):
        nonlocal nleaf
        if not records[i]['children']:xpos[i]=nleaf;nleaf+=1
        else:
            for j in records[i]['children']:place(j)
            xpos[i]=statistics.mean(xpos[j] for j in records[i]['children'])
    place(0); chosen={0};i=0
    while records[i]['children']:
        i=next(j for j in records[i]['children'] if abs(records[j]['value']-records[i]['value'])<1e-8);chosen.add(i)
    fig,ax=plt.subplots(figsize=(26,10))
    for a,b,label in edges:
        ax.plot([xpos[a],xpos[b]],[-records[a]['s'].depth,-records[b]['s'].depth],color=green if a in chosen and b in chosen else '#C8C8C8',lw=2.7 if a in chosen and b in chosen else .7,zorder=1)
    for node in records:
        s=node['s'];short=node['label'].replace('Programar con respaldo','Respaldo').replace('Posponer y monitorear','Posponer').replace('Programar solar','Solar').replace('Ventana 2 / ambiente','V2 amb.')
        tag='MAX' if s.depth%2==0 and s.depth<4 else ('MIN' if s.depth<4 else 'U')
        ax.text(xpos[node['id']],-s.depth,f"{short}\n{tag}: {node['value']:.2f}",ha='center',va='center',fontsize=6.4 if s.depth>=3 else 9,bbox=dict(boxstyle='round,pad=.35',fc=light if node['id'] in chosen else 'white',ec=green if node['id'] in chosen else '#999'),zorder=2)
    ax.set_ylim(-4.35,.5);ax.axis('off');ax.set_title('Minimax completo, profundidad 4 | rama elegida en verde | valores en puntos de política',fontsize=18)
    fig.text(.5,.02,'MIN usa escenarios históricos reales como pruebas de estrés. Las combinaciones de ventanas son contrafactuales, no pronósticos.',ha='center',fontsize=11)
    fig.tight_layout(rect=[0,.04,1,.98]);fig.savefig(out/'minmax_tree.png',dpi=200);plt.close(fig)
    #States graph = union of found candidate paths. Full predicates in each node.
    nodes={}; links=set(); selected_edges=set();selected_states=set()
    for action,plan in result['candidates'].items():
        prev=None
        for k,st in enumerate(plan['states']):
            key=tuple(st);nodes.setdefault(key,len(nodes))
            if prev is not None:
                link=(prev,key,plan['plan'][k-1]);links.add(link)
                if action==result['minimax']['decision']:selected_edges.add(link)
            if action==result['minimax']['decision']:selected_states.add(key)
            prev=key
    layers={}
    for key in nodes:
        depth=min(plan['states'].index(list(key)) for plan in result['candidates'].values() if list(key) in plan['states'])
        layers.setdefault(depth,[]).append(key)
    pos={key:(depth,-j+(len(keys)-1)/2) for depth,keys in layers.items() for j,key in enumerate(keys)}
    fig,ax=plt.subplots(figsize=(14,6.5))
    for a,b,label in sorted(links):
        x,y=pos[a];xx,yy=pos[b];sel=(a,b,label) in selected_edges
        ax.annotate('',xy=(xx-.15,yy),xytext=(x+.15,y),arrowprops={'arrowstyle':'->','color':green if sel else '#AAA','lw':2.7 if sel else .8})
        ax.text((x+xx)/2,(y+yy)/2+.30,label.replace(' ', '\n', 1),fontsize=9,ha='center',va='center',color=green if sel else '#666',bbox=dict(fc='white',ec='none',pad=1))
    for key,(x,y) in pos.items():
        label='S'+str(nodes[key])+'\n'+'\n'.join(key)
        ax.text(x,y,label,ha='center',va='center',fontsize=10,bbox=dict(boxstyle='round,pad=.45',fc=light if key in selected_states else '#FAFAFA',ec=green if key in selected_states else '#AAA'))
    ax.set_xlim(-.5,max(x for x,y in pos.values())+.6);ax.set_ylim(-1.15,1.15);ax.axis('off')
    ax.set_title('STRIPS | estados y planes candidatos | verde: plan final elegido',fontsize=17)
    fig.text(.5,.04,'Inicio S0: base, kit, pendiente. Metas: programación solar / con respaldo / espera. Con posterior < 0.60 se exige revisar sensores.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.08,1,1]);fig.savefig(out/'strips_graph.png',dpi=180);plt.close(fig)
    #Integrated summary genuinely uses the selected result and plan.
    fig,axs=plt.subplots(1,3,figsize=(16,7),gridspec_kw={'width_ratios':[1.2,.9,1.1]})
    axs[0].axis('off');axs[0].set_title('Plan elegido por STRIPS',fontweight='bold')
    for i,name in enumerate(result['selected_plan']['plan']):axs[0].text(.02,.84-i*.13,f'{i+1}. {name}',fontsize=15)
    axs[1].bar(['Apta','No apta'],[p,1-p],color=[green,red]);axs[1].set_ylim(0,1);axs[1].yaxis.set_major_formatter(PercentFormatter(1));axs[1].set_title('Bayes: ventana meteorológica')
    axs[1].text(0,min(.94,p+.03),f'{p:.1%}',ha='center',fontsize=20,fontweight='bold')
    axs[2].axis('off');axs[2].set_title('Decisión final Minimax',fontweight='bold');axs[2].text(.05,.69,LABELS[result['minimax']['decision']].replace(' con ','\ncon '),fontsize=23,color=green,fontweight='bold')
    axs[2].text(.05,.44,f"Peor valor: {result['minimax']['value']:.2f}\npuntos de política\nProfundidad 4 + poda α-β",fontsize=14)
    fig.suptitle(f"Agente agroenergético | fecha histórica {result['date']}",fontsize=20)
    fig.text(.5,.06,'Observar registros → inferir posterior → planificar alternativas → decidir → recomendar el plan seleccionado',ha='center',fontsize=12)
    fig.text(.5,.02,'No se ejecutan maniobras en equipos. Aptitud meteorológica no equivale a éxito físico del riego.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.12,1,.95]);fig.savefig(out/'agent_summary.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,5))
    ax.plot([s['prior'] for s in sensitivity],[s['posterior'] for s in sensitivity],marker='o',color=green)
    ax.axhline(POLICY['threshold'],color=red,ls='--',label='Umbral de revisión extra = 0.60')
    for s in sensitivity:ax.annotate(s['decision'],(s['prior'],s['posterior']),xytext=(3,8),textcoords='offset points',fontsize=8)
    ax.set(xlabel='Prior alternativo de R=1',ylabel='Posterior con la misma evidencia',ylim=(0,1),xlim=(0,1));ax.legend();fig.tight_layout();fig.savefig(out/'bayes_sensitivity.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--date');parser.add_argument('--prior',type=float);parser.add_argument('--output',default='results');parser.add_argument('--no-plots',action='store_true');parser.add_argument('--repeats',type=int,default=31)
    args=parser.parse_args()
    if args.repeats<1:raise ValueError('--repeats debe ser positivo.')
    out=BASE/args.output;out.mkdir(parents=True,exist_ok=True)
    rows,train,test,t,missing,raw=load_data();joint,cpt=fit_bayes(train);days=representative_days(train)
    #Three distinct evidence profiles, selected by min / middle / max posterior.
    unique={}
    for r in test:unique.setdefault((r['C'],r['E']),r)
    ordered=sorted(unique.values(),key=lambda r:posterior(joint,r['C'],r['E']))
    cases=[ordered[0],ordered[len(ordered)//2],ordered[-1]]
    record=next((r for r in test if r['date']==args.date),None) if args.date else cases[1]
    if record is None:raise ValueError('Fecha no válida: use un día de 2025 con día siguiente disponible.')
    result=flow(record,joint,days,t,args.prior)
    empirical_prior=sum(p for (c,e,r),p in joint.items() if r)
    sensitivity=[]
    for prior in sorted([.05,.20,.40,.60,.80,empirical_prior]):
        run=flow(record,joint,days,t,prior)
        sensitivity.append({'prior':prior,'posterior':run['posterior'],'goal':' + '.join(run['selected_plan']['goal']),
                            'plan':' -> '.join(run['selected_plan']['plan']),'steps':len(run['selected_plan']['plan']),
                            'decision':run['minimax']['decision'],'value':run['minimax']['value']})
    benches=benchmark(cases,joint,days,t,args.repeats)
    preds=[posterior(joint,r['C'],r['E']) for r in test]
    metrics={'brier':statistics.mean((p-r['R'])**2 for p,r in zip(preds,test)),
             'brier_prior_baseline':statistics.mean((empirical_prior-r['R'])**2 for r in test),
             'accuracy_05':statistics.mean((p>=.5)==bool(r['R']) for p,r in zip(preds,test)),
             'n_test':len(test),'R_test_prevalence':statistics.mean(r['R'] for r in test)}
    write_csv(BASE/'data/weather_daily.csv',rows);write_csv(BASE/'data/training_pairs.csv',train);write_csv(BASE/'data/test_pairs.csv',test)
    write_csv(out/'cpt_counts.csv',cpt);write_csv(out/'minimax_benchmark.csv',benches);write_csv(out/'sensitivity.csv',sensitivity)
    write_csv(out/'test_predictions.csv',[{**r,'predicted_R':p} for r,p in zip(test,preds)])
    report={'rows':len(rows),'train_pairs':len(train),'test_pairs':len(test),'missing':missing,'thresholds':t,'metrics':metrics,
            'prior':empirical_prior,'policy':POLICY,'historic_stress_days':days,'scenarios':cases,'run':result,'sensitivity':sensitivity}
    dump(out/'report.json',report)
    lines=[f'OBSERVE | fecha={record["date"]}; fuente=NASA POWER; radiación={record["solar_kwh_m2"]} kWh/m2/día; lluvia={record["rain_mm"]} mm/día; temperatura={record["temp_c"]} C',
           f'EVIDENCE | C(lluvia baja)={record["C"]}; E(radiación alta)={record["E"]}',
           f'POSTERIOR | P(ventana meteorológica apta mañana | evidencia)={result["posterior"]:.6f}',
           'NOTA | Es un evento derivado de registros ambientales; NO es éxito medido de riego.']
    for a,plan in result['candidates'].items():lines.append(f'PLAN CANDIDATO {a} | meta={plan["goal"]}; pasos={plan["plan"]}')
    lines += [f'DECISION | {LABELS[result["minimax"]["decision"]]}; Minimax alpha-beta={result["minimax"]["value"]:.6f}',
              'PLAN FINAL | '+' -> '.join(result['selected_plan']['plan']),
              'ACCIÓN | Recomendación registrada. No se ha accionado ningún equipo físico.',
              f'VALIDACIÓN | meta satisfecha en ejecución simbólica; Brier 2025={metrics["brier"]:.6f}; base={metrics["brier_prior_baseline"]:.6f}',
              f'DATOS | {len(rows)} días; entrenamiento={len(train)} pares; prueba={len(test)}; faltantes={len(missing)}',
              f'EXPERIMENTOS | {len(benches)} filas, 3 escenarios x 3 profundidades x 3 variantes; {args.repeats} repeticiones/mediana.',
              'SENSIBILIDAD | '+json.dumps(sensitivity,ensure_ascii=False)]
    log='\n'.join(lines);print(log);(out/'agent_log.txt').write_text(log+'\n')
    if not args.no_plots:charts(out,result,joint,days,t,sensitivity)
    print(f'Archivos guardados en {out}')

if __name__=='__main__':
    try:main()
    except (ValueError,OSError) as e:raise SystemExit(f'ERROR: {e}')
