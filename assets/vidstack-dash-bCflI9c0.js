var bt=Object.defineProperty;var Vt=(n,t,e)=>t in n?bt(n,t,{enumerable:!0,configurable:!0,writable:!0,value:e}):n[t]=e;var b=(n,t,e)=>(Vt(n,typeof t!="symbol"?t+"":t,e),e),dt=(n,t,e)=>{if(!t.has(n))throw TypeError("Cannot "+e)};var i=(n,t,e)=>(dt(n,t,"read from private field"),e?e.call(n):t.get(n)),r=(n,t,e)=>{if(t.has(n))throw TypeError("Cannot add the same private member more than once");t instanceof WeakSet?t.add(n):t.set(n,e)},l=(n,t,e,s)=>(dt(n,t,"write to private field"),s?s.call(n,e):t.set(n,e),e);var h=(n,t,e)=>(dt(n,t,"access private method"),e);import{e as Gt,v as $,P as Ut,p as Bt,l as ft,j as Kt,D as C,J as Jt,a1 as Yt,T as zt,M as Wt,a as St,aV as Xt,ag as Zt,r as kt}from"./vidstack-Ds_q5BGO-2OBULqdt.js";import{VideoProvider as ti}from"./vidstack-video-ibW5aN54.js";import{Q as yt,L as _}from"./vidstack-player-vx-017Wm.js";import{T as V,a as ii,c as ei}from"./vidstack-DXXgp8ue-2MT7yJzB.js";import{R as si}from"./vidstack-DSYpsFWk-v7Jbeavf.js";import"./vidstack-CGXAe0PE-PpCD5Q9_.js";import"./app-yiwvZYM8.js";function mt(n){try{return new Intl.DisplayNames(navigator.languages,{type:"language"}).of(n)??null}catch{return null}}const ni=n=>`dash-${kt(n)}`;var E,d,o,N,R,S,D,U,vt,B,wt,K,Et,v,F,J,Lt,Y,At,z,Dt,W,xt,X,Mt,Z,Ct,k,Nt,tt,Rt,L,it,Ft,P,G,q,ut,et,Pt,j,ct,st,_t,nt,$t,H,pt;class ri{constructor(t,e){r(this,S);r(this,U);r(this,B);r(this,K);r(this,J);r(this,Y);r(this,z);r(this,W);r(this,X);r(this,Z);r(this,k);r(this,tt);r(this,it);r(this,P);r(this,q);r(this,et);r(this,j);r(this,st);r(this,nt);r(this,H);r(this,E,void 0);r(this,d,void 0);r(this,o,null);r(this,N,new Set);r(this,R,null);b(this,"config",{});r(this,v,null);r(this,F,{});r(this,L,-1);l(this,E,t),l(this,d,e)}get instance(){return i(this,o)}setup(t){l(this,o,t().create());const e=h(this,K,Et).bind(this);for(const s of Object.values(t.events))i(this,o).on(s,e);i(this,o).on(t.events.ERROR,h(this,Z,Ct).bind(this));for(const s of i(this,N))s(i(this,o));i(this,d).player.dispatch("dash-instance",{detail:i(this,o)}),i(this,o).initialize(i(this,E),void 0,!1),i(this,o).updateSettings({streaming:{text:{defaultEnabled:!1,dispatchForManualRendering:!0},buffer:{fastSwitchEnabled:!0}},...this.config}),i(this,o).on(t.events.FRAGMENT_LOADING_STARTED,h(this,k,Nt).bind(this)),i(this,o).on(t.events.FRAGMENT_LOADING_COMPLETED,h(this,tt,Rt).bind(this)),i(this,o).on(t.events.MANIFEST_LOADED,h(this,X,Mt).bind(this)),i(this,o).on(t.events.QUALITY_CHANGE_RENDERED,h(this,W,xt).bind(this)),i(this,o).on(t.events.TEXT_TRACKS_ADDED,h(this,Y,At).bind(this)),i(this,o).on(t.events.TRACK_CHANGE_RENDERED,h(this,z,Dt).bind(this)),i(this,d).qualities[yt.enableAuto]=h(this,et,Pt).bind(this),ft(i(this,d).qualities,"change",h(this,st,_t).bind(this)),ft(i(this,d).audioTracks,"change",h(this,nt,$t).bind(this)),l(this,R,Kt(h(this,U,vt).bind(this)))}onInstance(t){return i(this,N).add(t),()=>i(this,N).delete(t)}loadSource(t){var e;h(this,H,pt).call(this),$(t.src)&&((e=i(this,o))==null||e.attachSource(t.src))}destroy(){var t,e;h(this,H,pt).call(this),(t=i(this,o))==null||t.destroy(),l(this,o,null),(e=i(this,R))==null||e.call(this),l(this,R,null)}}E=new WeakMap,d=new WeakMap,o=new WeakMap,N=new WeakMap,R=new WeakMap,S=new WeakSet,D=function(t){return new C(ni(t.type),{detail:t})},U=new WeakSet,vt=function(){if(!i(this,d).$state.live())return;const t=new si(h(this,B,wt).bind(this));return t.start(),t.stop.bind(t)},B=new WeakSet,wt=function(){if(!i(this,o))return;const t=i(this,o).duration()-i(this,o).time();i(this,d).$state.liveSyncPosition.set(isNaN(t)?1/0:t)},K=new WeakSet,Et=function(t){var e;(e=i(this,d).player)==null||e.dispatch(h(this,S,D).call(this,t))},v=new WeakMap,F=new WeakMap,J=new WeakSet,Lt=function(t){var u;const e=(u=i(this,v))==null?void 0:u[V.native],s=(e==null?void 0:e.track).cues;if(!e||!s)return;const p=i(this,v).id,g=i(this,F)[p]??0,f=h(this,S,D).call(this,t);for(let y=g;y<s.length;y++){const c=s[y];c.positionAlign||(c.positionAlign="auto"),i(this,v).addCue(c,f)}i(this,F)[p]=s.length},Y=new WeakSet,At=function(t){var g;if(!i(this,o))return;const e=t.tracks,s=[...i(this,E).textTracks].filter(f=>"manualMode"in f),p=h(this,S,D).call(this,t);for(let f=0;f<s.length;f++){const u=e[f],y=s[f],c=`dash-${u.kind}-${f}`,w=new ii({id:c,label:(u==null?void 0:u.label)??((g=u.labels.find(a=>a.text))==null?void 0:g.text)??((u==null?void 0:u.lang)&&mt(u.lang))??(u==null?void 0:u.lang)??void 0,language:u.lang??void 0,kind:u.kind,default:u.defaultTrack});w[V.native]={managed:!0,track:y},w[V.readyState]=2,w[V.onModeChange]=()=>{i(this,o)&&(w.mode==="showing"?(i(this,o).setTextTrack(f),l(this,v,w)):(i(this,o).setTextTrack(-1),l(this,v,null)))},i(this,d).textTracks.add(w,p)}},z=new WeakSet,Dt=function(t){const{mediaType:e,newMediaInfo:s}=t;if(e==="audio"){const p=i(this,d).audioTracks.getById(`dash-audio-${s.index}`);if(p){const g=h(this,S,D).call(this,t);i(this,d).audioTracks[_.select](p,!0,g)}}},W=new WeakSet,xt=function(t){if(t.mediaType!=="video")return;const e=i(this,d).qualities[t.newQuality];if(e){const s=h(this,S,D).call(this,t);i(this,d).qualities[_.select](e,!0,s)}},X=new WeakSet,Mt=function(t){if(i(this,d).$state.canPlay()||!i(this,o))return;const{type:e,mediaPresentationDuration:s}=t.data,p=h(this,S,D).call(this,t);i(this,d).notify("stream-type-change",e!=="static"?"live":"on-demand",p),i(this,d).notify("duration-change",s,p),i(this,d).qualities[yt.setAuto](!0,p);const g=i(this,o).getVideoElement(),f=i(this,o).getTracksForTypeFromManifest("video",t.data),u=[...new Set(f.map(a=>a.mimeType))].find(a=>a&&Jt(g,a)),y=f.filter(a=>u===a.mimeType)[0];let c=i(this,o).getTracksForTypeFromManifest("audio",t.data);const w=[...new Set(c.map(a=>a.mimeType))].find(a=>a&&Yt(g,a));if(c=c.filter(a=>w===a.mimeType),y.bitrateList.forEach((a,Q)=>{var M;const lt={id:((M=a.id)==null?void 0:M.toString())??`dash-bitrate-${Q}`,width:a.width??0,height:a.height??0,bitrate:a.bandwidth??0,codec:y.codec,index:Q};i(this,d).qualities[_.add](lt,p)}),zt(y.index)){const a=i(this,d).qualities[y.index];a&&i(this,d).qualities[_.select](a,!0,p)}c.forEach((a,Q)=>{const M=a.labels.find(gt=>navigator.languages.some(Qt=>gt.lang&&Qt.toLowerCase().startsWith(gt.lang.toLowerCase())))||a.labels[0],Ot={id:`dash-audio-${a==null?void 0:a.index}`,label:(M==null?void 0:M.text)??(a.lang&&mt(a.lang))??a.lang??"",language:a.lang??"",kind:"main",mimeType:a.mimeType,codec:a.codec,index:Q};i(this,d).audioTracks[_.add](Ot,p)}),g.dispatchEvent(new C("canplay",{trigger:p}))},Z=new WeakSet,Ct=function(t){const{type:e,error:s}=t;switch(s.code){case 27:h(this,it,Ft).call(this,s);break;default:h(this,q,ut).call(this,s);break}},k=new WeakSet,Nt=function(){i(this,L)>=0&&h(this,P,G).call(this)},tt=new WeakSet,Rt=function(t){t.mediaType==="text"&&requestAnimationFrame(h(this,J,Lt).bind(this,t))},L=new WeakMap,it=new WeakSet,Ft=function(t){var e;h(this,P,G).call(this),(e=i(this,o))==null||e.play(),l(this,L,window.setTimeout(()=>{l(this,L,-1),h(this,q,ut).call(this,t)},5e3))},P=new WeakSet,G=function(){clearTimeout(i(this,L)),l(this,L,-1)},q=new WeakSet,ut=function(t){i(this,d).notify("error",{message:t.message??"",code:1,error:t})},et=new WeakSet,Pt=function(){var e;h(this,j,ct).call(this,"video",!0);const{qualities:t}=i(this,d);(e=i(this,o))==null||e.setQualityFor("video",t.selectedIndex,!0)},j=new WeakSet,ct=function(t,e){var s;(s=i(this,o))==null||s.updateSettings({streaming:{abr:{autoSwitchBitrate:{[t]:e}}}})},st=new WeakSet,_t=function(){const{qualities:t}=i(this,d);!i(this,o)||t.auto||!t.selected||(h(this,j,ct).call(this,"video",!1),i(this,o).setQualityFor("video",t.selectedIndex,t.switch==="current"),Wt&&(i(this,E).currentTime=i(this,E).currentTime))},nt=new WeakSet,$t=function(){if(!i(this,o))return;const{audioTracks:t}=i(this,d),e=i(this,o).getTracksFor("audio").find(s=>t.selected&&t.selected.id===`dash-audio-${s.index}`);e&&i(this,o).setCurrentTrack(e)},H=new WeakSet,pt=function(){h(this,P,G).call(this),l(this,v,null),l(this,F,{})};var x,T,I,rt,qt,ot,jt,at,Ht,ht,It;class oi{constructor(t,e,s){r(this,rt);r(this,ot);r(this,at);r(this,ht);r(this,x,void 0);r(this,T,void 0);r(this,I,void 0);l(this,x,t),l(this,T,e),l(this,I,s),h(this,rt,qt).call(this)}}x=new WeakMap,T=new WeakMap,I=new WeakMap,rt=new WeakSet,qt=async function(){const t={onLoadStart:h(this,ot,jt).bind(this),onLoaded:h(this,at,Ht).bind(this),onLoadError:h(this,ht,It).bind(this)};let e=await hi(i(this,x),t);if(St(e)&&!$(i(this,x))&&(e=await ai(i(this,x),t)),!e)return null;if(!window.dashjs.supportsMediaSource()){const s="[vidstack] `dash.js` is not supported in this environment";return i(this,T).player.dispatch(new C("dash-unsupported")),i(this,T).notify("error",{message:s,code:4}),null}return e},ot=new WeakSet,jt=function(){i(this,T).player.dispatch(new C("dash-lib-load-start"))},at=new WeakSet,Ht=function(t){i(this,T).player.dispatch(new C("dash-lib-loaded",{detail:t})),i(this,I).call(this,t)},ht=new WeakSet,It=function(t){const e=ei(t);i(this,T).player.dispatch(new C("dash-lib-load-error",{detail:e})),i(this,T).notify("error",{message:e.message,code:4,error:e})};async function ai(n,t={}){var e,s,p,g,f,u,y;if(!St(n)){if((e=t.onLoadStart)==null||e.call(t),di(n))return(s=t.onLoaded)==null||s.call(t,n),n;if(Tt(n)){const c=n.MediaPlayer;return(p=t.onLoaded)==null||p.call(t,c),c}try{const c=(g=await n())==null?void 0:g.default;if(Tt(c))return(f=t.onLoaded)==null||f.call(t,c.MediaPlayer),c.MediaPlayer;if(c)(u=t.onLoaded)==null||u.call(t,c);else throw Error("");return c}catch(c){(y=t.onLoadError)==null||y.call(t,c)}}}async function hi(n,t={}){var e,s,p;if($(n)){(e=t.onLoadStart)==null||e.call(t);try{if(await Xt(n),!Zt(window.dashjs.MediaPlayer))throw Error("");const g=window.dashjs.MediaPlayer;return(s=t.onLoaded)==null||s.call(t,g),g}catch(g){(p=t.onLoadError)==null||p.call(t,g)}}}function di(n){return n&&n.prototype&&n.prototype!==Function}function Tt(n){return n&&"MediaPlayer"in n}const ui="https://cdn.jsdelivr.net";var O,m,A;class ci extends ti{constructor(){super(...arguments);b(this,"$$PROVIDER_TYPE","DASH");r(this,O,null);r(this,m,new ri(this.video,this.ctx));r(this,A,`${ui}/npm/dashjs@4.7.4/dist/dash.all.min.js`)}get ctor(){return i(this,O)}get instance(){return i(this,m).instance}get type(){return"dash"}get canLiveSync(){return!0}get config(){return i(this,m).config}set config(e){i(this,m).config=e}get library(){return i(this,A)}set library(e){l(this,A,e)}preconnect(){$(i(this,A))&&Ut(i(this,A))}setup(){super.setup(),new oi(i(this,A),this.ctx,e=>{l(this,O,e),i(this,m).setup(e),this.ctx.notify("provider-setup",this);const s=Bt(this.ctx.$state.source);s&&this.loadSource(s)})}async loadSource(e,s){if(!$(e.src)){this.removeSource();return}this.media.preload=s||"",this.appendSource(e,"application/x-mpegurl"),i(this,m).loadSource(e),this.currentSrc=e}onInstance(e){const s=i(this,m).instance;return s&&e(s),i(this,m).onInstance(e)}destroy(){i(this,m).destroy()}}O=new WeakMap,m=new WeakMap,A=new WeakMap,b(ci,"supported",Gt());export{ci as DASHProvider};