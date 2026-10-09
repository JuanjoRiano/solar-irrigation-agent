#Verifica datos conservados, separación temporal y coherencia de los tres métodos.
import hashlib,json,math,unittest
from itertools import product
import agent as a

class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows,cls.train,cls.test,cls.t,_,_=a.load_data()
        cls.joint,_=a.fit_bayes(cls.train)
        cls.days=a.representative_days(cls.train)

    def test_source_and_temporal_split(self):
        manifest=json.loads((a.BASE/'data/provenance.json').read_text())
        self.assertEqual(hashlib.sha256((a.BASE/'data/nasa_power_raw.json').read_bytes()).hexdigest(),manifest['raw_sha256'])
        self.assertEqual((len(self.rows),len(self.train),len(self.test)),(2192,1826,364))
        train_dates={x[k] for x in self.train for k in ('date','next_date')}
        test_dates={x[k] for x in self.test for k in ('date','next_date')}
        self.assertFalse(train_dates & test_dates)
        for day in self.days:self.assertIn(day,self.train)

    def test_bayes_and_prior_identity(self):
        self.assertAlmostEqual(sum(self.joint.values()),1.)
        prior=sum(v for (c,e,r),v in self.joint.items() if r)
        for c,e in product((0,1),repeat=2):
            p=a.posterior(self.joint,c,e)
            self.assertTrue(0<=p<=1)
            self.assertAlmostEqual(a.posterior(self.joint,c,e,prior),p)
        self.assertAlmostEqual(a.posterior(self.joint,1,0),(132+1)/(342+2))

    def test_alpha_beta_and_exhaustive_oracle(self):
        #Todos los perfiles C/E, los tres horizontes y probabilidades extremas.
        ps=[a.posterior(self.joint,c,e) for c,e in product((0,1),repeat=2)]+[0.,1.]
        for p in ps:
            for depth in (2,3,4):
                n=a.minimax(p,self.days,self.t,depth,'naive')
                ab=a.minimax(p,self.days,self.t,depth,'alpha_beta')
                self.assertEqual(n['decision'],ab['decision']);self.assertAlmostEqual(n['value'],ab['value'])
            full=a.minimax(p,self.days,self.t)
            oracle={action:a.exact_action_value(action,p,self.days,self.t) for action in a.ACTIONS}
            self.assertAlmostEqual(full['value'],max(oracle.values()))
            self.assertEqual(full['visited'],67);self.assertEqual(full['evaluated'],36)
            
    def test_plans_and_sensitivity_integration(self):
        record=next(x for x in self.test if x['date']=='2025-01-24')
        low=a.flow(record,self.joint,self.days,self.t,.05)
        base=a.flow(record,self.joint,self.days,self.t)
        high=a.flow(record,self.joint,self.days,self.t,.6)
        self.assertEqual(low['minimax']['decision'],'posponer')
        self.assertEqual(base['minimax']['decision'],'solar')
        self.assertEqual(high['minimax']['decision'],'solar')
        self.assertEqual([len(x['selected_plan']['plan']) for x in (low,base,high)],[3,4,3])
        for run in (low,base,high):
            for plan in run['candidates'].values():a.validate_plan(plan)
            self.assertEqual(run['selected_plan'],run['candidates'][run['minimax']['decision']])
        self.assertIn('Mantener',base['minimax']['path'])

if __name__=='__main__':unittest.main(verbosity=2)
