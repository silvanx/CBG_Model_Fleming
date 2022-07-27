/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "scoplib.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__GABAa_S
#define _nrn_initial _nrn_initial__GABAa_S
#define nrn_cur _nrn_cur__GABAa_S
#define _nrn_current _nrn_current__GABAa_S
#define nrn_jacob _nrn_jacob__GABAa_S
#define nrn_state _nrn_state__GABAa_S
#define _net_receive _net_receive__GABAa_S 
#define release release__GABAa_S 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define Cmax _p[0]
#define i _p[1]
#define g _p[2]
#define C _p[3]
#define R _p[4]
#define R0 _p[5]
#define Rinf _p[6]
#define Rtau _p[7]
#define on _p[8]
#define gmax _p[9]
#define tLast _p[10]
#define nspike _p[11]
#define collisionBlock _p[12]
#define v _p[13]
#define _g _p[14]
#define _tsav _p[15]
#define _nd_area  *_ppvar[0]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 /* declaration of user functions */
 static double _hoc_exptable(void*);
 static double _hoc_release(void*);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern Prop* nrn_point_prop_;
 static int _pointtype;
 static void* _hoc_create_pnt(Object* _ho) { void* create_point_process(int, Object*);
 return create_point_process(_pointtype, _ho);
}
 static void _hoc_destroy_pnt(void*);
 static double _hoc_loc_pnt(void* _vptr) {double loc_point_process(int, void*);
 return loc_point_process(_pointtype, _vptr);
}
 static double _hoc_has_loc(void* _vptr) {double has_loc_point(void*);
 return has_loc_point(_vptr);
}
 static double _hoc_get_loc_pnt(void* _vptr) {
 double get_loc_point_process(void*); return (get_loc_point_process(_vptr));
}
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata(void* _vptr) { Prop* _prop;
 _prop = ((Point_process*)_vptr)->_prop;
   _setdata(_prop);
 }
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 0,0
};
 static Member_func _member_func[] = {
 "loc", _hoc_loc_pnt,
 "has_loc", _hoc_has_loc,
 "get_loc", _hoc_get_loc_pnt,
 "exptable", _hoc_exptable,
 "release", _hoc_release,
 0, 0
};
#define _f_exptable _f_exptable_GABAa_S
#define exptable exptable_GABAa_S
 extern double _f_exptable( _threadargsprotocomma_ double );
 extern double exptable( _threadargsprotocomma_ double );
 
static void _check_exptable(double*, Datum*, Datum*, NrnThread*); 
static void _check_table_thread(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, int _type) {
   _check_exptable(_p, _ppvar, _thread, _nt);
 }
 /* declare global and static user variables */
#define Alpha Alpha_GABAa_S
 double Alpha = 10.5;
#define Beta Beta_GABAa_S
 double Beta = 0.166;
#define Cdur Cdur_GABAa_S
 double Cdur = 0.3;
#define Erev Erev_GABAa_S
 double Erev = -80;
#define blockTime blockTime_GABAa_S
 double blockTime = 2;
#define usetable usetable_GABAa_S
 double usetable = 1;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "usetable_GABAa_S", 0, 1,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "Cdur_GABAa_S", "ms",
 "Alpha_GABAa_S", "/ms",
 "Beta_GABAa_S", "/ms",
 "Erev_GABAa_S", "mV",
 "blockTime_GABAa_S", "ms",
 "Cmax", "mM",
 "i", "nA",
 "g", "umho",
 "C", "mM",
 0,0
};
 static double delta_t = 0.01;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "Cdur_GABAa_S", &Cdur_GABAa_S,
 "Alpha_GABAa_S", &Alpha_GABAa_S,
 "Beta_GABAa_S", &Beta_GABAa_S,
 "Erev_GABAa_S", &Erev_GABAa_S,
 "blockTime_GABAa_S", &blockTime_GABAa_S,
 "usetable_GABAa_S", &usetable_GABAa_S,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(NrnThread*, _Memb_list*, int);
static void nrn_state(NrnThread*, _Memb_list*, int);
 static void nrn_cur(NrnThread*, _Memb_list*, int);
static void  nrn_jacob(NrnThread*, _Memb_list*, int);
 static void _hoc_destroy_pnt(void* _vptr) {
   destroy_point_process(_vptr);
}
 
static int _ode_count(int);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"GABAa_S",
 "Cmax",
 0,
 "i",
 "g",
 "C",
 "R",
 "R0",
 0,
 0,
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
  if (nrn_point_prop_) {
	_prop->_alloc_seq = nrn_point_prop_->_alloc_seq;
	_p = nrn_point_prop_->param;
	_ppvar = nrn_point_prop_->dparam;
 }else{
 	_p = nrn_prop_data_alloc(_mechtype, 16, _prop);
 	/*initialize range parameters*/
 	Cmax = 0.5;
  }
 	_prop->param = _p;
 	_prop->param_size = 16;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 3, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
 
#define _tqitem &(_ppvar[2]._pvoid)
 static void _net_receive(Point_process*, double*, double);
 static void _net_init(Point_process*, double*, double);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _Destexhe_Static_GABAA_Synapse_reg() {
	int _vectorized = 1;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex, 1,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_table_reg(_mechtype, _check_table_thread);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 16, 3);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "netsend");
 	hoc_register_cvode(_mechtype, _ode_count, 0, 0, 0);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_init[_mechtype] = _net_init;
 pnt_receive_size[_mechtype] = 3;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 GABAa_S Destexhe_Static_GABAA_Synapse.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double *_t_exptable;
static int _reset;
static char *modelname = "simple GABA-A receptors, based on AMPA model (discrete connections)";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int release(_threadargsproto_);
 static double _n_exptable(_threadargsprotocomma_ double _lv);
 
static int  release ( _threadargsproto_ ) {
   if ( on ) {
     R = gmax * Rinf + ( R0 - gmax * Rinf ) * exptable ( _threadargscomma_ - ( t - tLast ) / Rtau ) ;
     }
   else {
     R = R0 * exptable ( _threadargscomma_ - Beta * ( t - tLast ) ) ;
     }
    return 0; }
 
static double _hoc_release(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r = 1.;
 release ( _p, _ppvar, _thread, _nt );
 return(_r);
}
 
static void _net_receive (Point_process* _pnt, double* _args, double _lflag) 
{  double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _thread = (Datum*)0; _nt = (NrnThread*)_pnt->_vnt;   _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ extern char* hoc_object_name(); hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t;   if (_lflag == 1. ) {*(_tqitem) = 0;}
 {
   double _lok , _ltmp ;
 if ( _lflag  == 0.0 ) {
     _lok = 0.0 ;
     if ( _args[1]  == 1.0 ) {
       collisionBlock = collisionBlock + 1.0 ;
       net_send ( _tqitem, _args, _pnt, t +  blockTime , - 1.0 ) ;
       _lok = 1.0 ;
       }
     else if ( collisionBlock  == 0.0 ) {
       _lok = 1.0 ;
       }
     if ( _lok ) {
       if (  ! on ) {
         on = 1.0 ;
         tLast = t ;
         R0 = R ;
         gmax = _args[0] ;
         }
       nspike = nspike + 1.0 ;
       net_send ( _tqitem, _args, _pnt, t +  Cdur , nspike ) ;
       }
     }
   else if ( _lflag  == nspike ) {
     if ( on ) {
       on = 0.0 ;
       tLast = t ;
       R0 = R ;
       gmax = 0.0 ;
       }
     }
   else if ( _lflag  == - 1.0 ) {
     collisionBlock = collisionBlock - 1.0 ;
     }
   } }
 
static void _net_init(Point_process* _pnt, double* _args, double _lflag) {
       double* _p = _pnt->_prop->param;
    Datum* _ppvar = _pnt->_prop->dparam;
    Datum* _thread = (Datum*)0;
    NrnThread* _nt = (NrnThread*)_pnt->_vnt;
 }
 static double _mfac_exptable, _tmin_exptable;
  static void _check_exptable(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  static int _maktable=1; int _i, _j, _ix = 0;
  double _xi, _tmax;
  if (!usetable) {return;}
  if (_maktable) { double _x, _dx; _maktable=0;
   _tmin_exptable =  - 10.0 ;
   _tmax =  10.0 ;
   _dx = (_tmax - _tmin_exptable)/2000.; _mfac_exptable = 1./_dx;
   for (_i=0, _x=_tmin_exptable; _i < 2001; _x += _dx, _i++) {
    _t_exptable[_i] = _f_exptable(_p, _ppvar, _thread, _nt, _x);
   }
  }
 }

 double exptable(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _lx) { 
#if 0
_check_exptable(_p, _ppvar, _thread, _nt);
#endif
 return _n_exptable(_p, _ppvar, _thread, _nt, _lx);
 }

 static double _n_exptable(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _lx){ int _i, _j;
 double _xi, _theta;
 if (!usetable) {
 return _f_exptable(_p, _ppvar, _thread, _nt, _lx); 
}
 _xi = _mfac_exptable * (_lx - _tmin_exptable);
 if (isnan(_xi)) {
  return _xi; }
 if (_xi <= 0.) {
 return _t_exptable[0];
 }
 if (_xi >= 2000.) {
 return _t_exptable[2000];
 }
 _i = (int) _xi;
 return _t_exptable[_i] + (_xi - (double)_i)*(_t_exptable[_i+1] - _t_exptable[_i]);
 }

 
double _f_exptable ( _threadargsprotocomma_ double _lx ) {
   double _lexptable;
 if ( ( _lx > - 10.0 )  && ( _lx < 10.0 ) ) {
     _lexptable = exp ( _lx ) ;
     }
   else {
     _lexptable = 0. ;
     }
   
return _lexptable;
 }
 
static double _hoc_exptable(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 
#if 1
 _check_exptable(_p, _ppvar, _thread, _nt);
#endif
 _r =  exptable ( _p, _ppvar, _thread, _nt, *getarg(1) );
 return(_r);
}
 
static int _ode_count(int _type){ hoc_execerror("GABAa_S", "cannot be used with CVODE"); return 0;}

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
 {
   R = 0.0 ;
   C = 0.0 ;
   Rinf = Cmax * Alpha / ( Cmax * Alpha + Beta ) ;
   Rtau = 1.0 / ( ( Alpha * Cmax ) + Beta ) ;
   on = 0.0 ;
   R0 = 0.0 ;
   nspike = 0.0 ;
   collisionBlock = 0.0 ;
   }
 
}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];

#if 0
 _check_exptable(_p, _ppvar, _thread, _nt);
#endif
 _tsav = -1e20;
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
 initmodel(_p, _ppvar, _thread, _nt);
}
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   i = R * ( v - Erev ) ;
   }
 _current += i;

} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
 	}
 _g = (_g - _rhs)/.001;
 _g *=  1.e2/(_nd_area);
 _rhs *= 1.e2/(_nd_area);
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}
 
}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}
 
}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
 {  { release(_p, _ppvar, _thread, _nt); }
  }}}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
   _t_exptable = makevector(2001*sizeof(double));
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "Destexhe_Static_GABAA_Synapse.mod";
static const char* nmodl_file_text = 
  "TITLE simple GABA-A receptors, based on AMPA model (discrete connections)\n"
  "\n"
  "COMMENT\n"
  "-----------------------------------------------------------------------------\n"
  "\n"
  "	Simple model for glutamate AMPA receptors\n"
  "	=========================================\n"
  "\n"
  "  - FIRST-ORDER KINETICS, FIT TO WHOLE-CELL RECORDINGS\n"
  "\n"
  "    Whole-cell recorded postsynaptic currents mediated by AMPA/Kainate\n"
  "    receptors (Xiang et al., J. Neurophysiol. 71: 2552-2556, 1994) were used\n"
  "    to estimate the parameters of the present model; the fit was performed\n"
  "    using a simplex algorithm (see Destexhe et al., J. Computational Neurosci.\n"
  "    1: 195-230, 1994).\n"
  "\n"
  "  - SHORT PULSES OF TRANSMITTER (0.3 ms, 0.5 mM)\n"
  "\n"
  "    The simplified model was obtained from a detailed synaptic model that \n"
  "    included the release of transmitter in adjacent terminals, its lateral \n"
  "    diffusion and uptake, and its binding on postsynaptic receptors (Destexhe\n"
  "    and Sejnowski, 1995).  Short pulses of transmitter with first-order\n"
  "    kinetics were found to be the best fast alternative to represent the more\n"
  "    detailed models.\n"
  "\n"
  "  - ANALYTIC EXPRESSION\n"
  "\n"
  "    The first-order model can be solved analytically, leading to a very fast\n"
  "    mechanism for simulating synapses, since no differential equation must be\n"
  "    solved (see references below).\n"
  "\n"
  "\n"
  "\n"
  "References\n"
  "\n"
  "   Destexhe, A., Mainen, Z.F. and Sejnowski, T.J.  An efficient method for\n"
  "   computing synaptic conductances based on a kinetic model of receptor binding\n"
  "   Neural Computation 6: 10-14, 1994.  \n"
  "\n"
  "   Destexhe, A., Mainen, Z.F. and Sejnowski, T.J. Synthesis of models for\n"
  "   excitable membranes, synaptic transmission and neuromodulation using a \n"
  "   common kinetic formalism, Journal of Computational Neuroscience 1: \n"
  "   195-230, 1994.\n"
  "\n"
  "See also:\n"
  "\n"
  "   http://www.cnl.salk.edu/~alain\n"
  "   http://cns.fmed.ulaval.ca\n"
  "\n"
  "-----------------------------------------------------------------------------\n"
  "ENDCOMMENT\n"
  "\n"
  "\n"
  "\n"
  ":INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}\n"
  "\n"
  "NEURON {\n"
  "	POINT_PROCESS GABAa_S\n"
  "	RANGE C, R, R0, R1, g, Cmax\n"
  "	NONSPECIFIC_CURRENT i\n"
  "	:GLOBAL Cdur, Alpha, Beta, Erev, blockTime, Rinf, Rtau\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "	(nA) = (nanoamp)\n"
  "	(mV) = (millivolt)\n"
  "	(umho) = (micromho)\n"
  "	(mM) = (milli/liter)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "\n"
  "	Cmax	= 0.5	(mM)		: max transmitter concentration	(set = 1 to match ~/netcon/ampa.hoc)\n"
  "	Cdur	= 0.3	(ms)		: transmitter duration (rising phase)\n"
  "	Alpha	= 10.5	(/ms mM)	: forward (binding) rate\n"
  "	Beta	= 0.166	(/ms)		: backward (unbinding) rate\n"
  "	Erev	= -80	(mV)		: reversal potential\n"
  "	blockTime = 2	(ms)		: time window following dbs event during which non-dbs events are blocked\n"
  "}\n"
  "\n"
  "\n"
  "ASSIGNED {\n"
  "	v		(mV)		: postsynaptic voltage\n"
  "	i 		(nA)		: current = g*(v - Erev)\n"
  "	g 		(umho)		: conductance\n"
  "	C		(mM)		: transmitter concentration\n"
  "	R					: fraction of open channels\n"
  "	R0					: open channels at start of time period\n"
  "	Rinf				: steady state channels open\n"
  "	Rtau		(ms)	: time constant of channel binding\n"
  "	on					: rising phase of PSC\n"
  "	gmax				: max conductance\n"
  "	tLast\n"
  "	nspike\n"
  "	collisionBlock\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "	R = 0\n"
  "	C = 0\n"
  "	Rinf = Cmax*Alpha / (Cmax*Alpha + Beta)\n"
  "	Rtau = 1 / ((Alpha * Cmax) + Beta)\n"
  "	on = 0\n"
  "	R0 = 0\n"
  "	nspike = 0\n"
  "	collisionBlock = 0\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE release\n"
  "	i = R*(v - Erev)\n"
  "}\n"
  "\n"
  "PROCEDURE release() {\n"
  "\n"
  "\n"
  "	if (on) {				: transmitter being released?\n"
  "\n"
  "	   R = gmax*Rinf + (R0 - gmax*Rinf) * exptable (- (t - tLast) / Rtau)\n"
  "				\n"
  "	} else {				: no release occuring\n"
  "\n"
  "  	   R = R0 * exptable (- Beta * (t - tLast))\n"
  "	}\n"
  "\n"
  "}\n"
  "\n"
  "\n"
  ": following supports both saturation from single input and\n"
  ": summation from multiple inputs\n"
  ": if spike occurs during CDur then new off time is t + CDur\n"
  ": ie. transmitter concatenates but does not summate\n"
  ": Note: automatic initialization of all reference args to 0 except first\n"
  "\n"
  "NET_RECEIVE(weight, ncType, ncPrb) {LOCAL ok, tmp\n"
  "\n"
  "	:ncType 0=presyn cell, 1=dbs activated axon\n"
  "	:MOVED TO dbsStim.mod 4/11/07		ncPrb probability that incoming event causes PSP\n"
  "\n"
  "	INITIAL {\n"
  "	}\n"
  "\n"
  "	: flag is an implicit argument of NET_RECEIVE and  normally 0\n"
  "      if (flag == 0) { : a spike, so turn on if not already in a Cdur pulse\n"
  "		ok = 0\n"
  "\n"
  "		if (ncType == 1) {\n"
  "			collisionBlock = collisionBlock + 1\n"
  "			net_send(blockTime, -1)\n"
  "\n"
  ":			tmp = scop_random()\n"
  ":			if (tmp <= ncPrb) {\n"
  ":				ok = 1\n"
  ":			}\n"
  "			ok = 1\n"
  "\n"
  "		}\n"
  "		else \n"
  "		if (collisionBlock == 0) {\n"
  "			ok = 1\n"
  "		}\n"
  "\n"
  "		if (ok) {\n"
  "			if (!on) {\n"
  "				on = 1\n"
  "				tLast = t\n"
  "				R0 = R\n"
  "				gmax = weight	:weight not additive from separate sources as in original ampa.mod\n"
  "			}\n"
  "\n"
  "			nspike = nspike + 1\n"
  "			: come again in Cdur with flag = current value of nspike\n"
  "			net_send(Cdur, nspike)			\n"
  "		}\n"
  "      }\n"
  "	else\n"
  "	if (flag == nspike) { : if this associated with last spike then turn off\n"
  "		if (on) {\n"
  "			on = 0\n"
  "			tLast = t\n"
  "			R0 = R\n"
  "			gmax = 0\n"
  "		}\n"
  "	} \n"
  "	else\n"
  "	if (flag == -1) {\n"
  "		collisionBlock = collisionBlock - 1\n"
  "	}\n"
  "\n"
  "\n"
  "}\n"
  "\n"
  "\n"
  "FUNCTION exptable(x) { \n"
  "	TABLE  FROM -10 TO 10 WITH 2000\n"
  "\n"
  "	if ((x > -10) && (x < 10)) {\n"
  "		exptable = exp(x)\n"
  "	} else {\n"
  "		exptable = 0.\n"
  "	}\n"
  "}\n"
  ;
#endif
