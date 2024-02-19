import{o as i,e as _,f as e,d as A,b as $,p as k,q as h,s as C,_ as q,v as y,w as F,x as I,c as N,a as R,y as U,z as M,A as E,B as j,C as O,D as W,E as Z,G,H as X,I as J,J as K,K as Q,L as Y,M as ee,g as t,N as te,l as o,t as se,n as v,i as P,O as V,S as B,m as S,P as oe,Q as H,R as T,T as ne,j as le,U as b,V as ae,W as ie,F as re,X as ce,Y as ue,Z as de,$ as L,a0 as _e,a1 as pe,a2 as me,a3 as ve}from"./index-1Fy_vBg_.js";import{N as he}from"./NoteDisplay-ihuOSRBM.js";import fe from"./DrawingControls-jrCt2BUB.js";const ge={class:"slidev-icon",viewBox:"0 0 32 32",width:"1.2em",height:"1.2em"},xe=e("path",{fill:"currentColor",d:"M8 12h10v2H8z"},null,-1),we=e("path",{fill:"currentColor",d:"M21.448 20A10.856 10.856 0 0 0 24 13a11 11 0 1 0-11 11a10.856 10.856 0 0 0 7-2.552L27.586 29L29 27.586ZM13 22a9 9 0 1 1 9-9a9.01 9.01 0 0 1-9 9"},null,-1),ye=[xe,we];function Se(l,r){return i(),_("svg",ge,[...ye])}const be={name:"carbon-zoom-out",render:Se},$e={class:"slidev-icon",viewBox:"0 0 32 32",width:"1.2em",height:"1.2em"},ke=e("path",{fill:"currentColor",d:"M18 12h-4V8h-2v4H8v2h4v4h2v-4h4z"},null,-1),Ce=e("path",{fill:"currentColor",d:"M21.448 20A10.856 10.856 0 0 0 24 13a11 11 0 1 0-11 11a10.856 10.856 0 0 0 7-2.552L27.586 29L29 27.586ZM13 22a9 9 0 1 1 9-9a9.01 9.01 0 0 1-9 9"},null,-1),ze=[ke,Ce];function Ne(l,r){return i(),_("svg",$e,[...ze])}const Me={name:"carbon-zoom-in",render:Ne},Pe={class:"slidev-icon",viewBox:"0 0 32 32",width:"1.2em",height:"1.2em"},Ve=e("path",{fill:"currentColor",d:"M12 10H6.78A11 11 0 0 1 27 16h2A13 13 0 0 0 6 7.68V4H4v8h8zm8 12h5.22A11 11 0 0 1 5 16H3a13 13 0 0 0 23 8.32V28h2v-8h-8z"},null,-1),Be=[Ve];function He(l,r){return i(),_("svg",Pe,[...Be])}const Te={name:"carbon-renew",render:He},Le={class:"slidev-icon",viewBox:"0 0 32 32",width:"1.2em",height:"1.2em"},Ae=e("path",{fill:"currentColor",d:"M16 30a14 14 0 1 1 14-14a14 14 0 0 1-14 14m0-26a12 12 0 1 0 12 12A12 12 0 0 0 16 4"},null,-1),qe=e("path",{fill:"currentColor",d:"M20.59 22L15 16.41V7h2v8.58l5 5.01z"},null,-1),De=[Ae,qe];function Fe(l,r){return i(),_("svg",Le,[...De])}const Ie={name:"carbon-time",render:Fe},Re="/logseq-faiss-chatgpt/assets/logo-title-horizontal-XSaaVPPu.png",Ue=A({__name:"NoteStatic",props:{class:{type:String,required:!1}},setup(l){const r=l,g=$(()=>{var c,u,n;return(n=(u=(c=k.value)==null?void 0:c.meta)==null?void 0:u.slide)==null?void 0:n.note}),x=$(()=>{var c,u,n;return(n=(u=(c=k.value)==null?void 0:c.meta)==null?void 0:u.slide)==null?void 0:n.noteHTML});return(c,u)=>(i(),h(he,{class:C(r.class),note:g.value,"note-html":x.value},null,8,["class","note","note-html"]))}}),Ee=q(Ue,[["__file","/Users/martin/Projects/pugs/presentations/logseq-faiss-chatgpt/slidev/node_modules/@slidev/client/internals/NoteStatic.vue"]]),f=l=>(me("data-v-574fd206"),l=l(),ve(),l),je={class:"bg-main h-full slidev-presenter"},Oe={class:"grid-section top flex"},We=f(()=>e("img",{src:Re,class:"ml-2 my-auto h-10 py-1 lg:h-14 lg:py-2",style:{height:"3.5rem"},alt:"Slidev logo"},null,-1)),Ze=f(()=>e("div",{class:"flex-auto"},null,-1)),Ge={class:"text-2xl pl-2 pr-6 my-auto tabular-nums"},Xe=f(()=>e("div",{class:"context"}," current ",-1)),Je=f(()=>e("div",{class:"context"}," next ",-1)),Ke={key:1,class:"grid-section note grid grid-rows-[1fr_min-content] overflow-hidden"},Qe={class:"border-t border-main py-1 px-2 text-sm"},Ye={class:"grid-section bottom"},et={class:"progress-bar"},tt=A({__name:"Presenter",setup(l){const r=y();F(),I(r);const g=N.titleTemplate.replace("%s",N.title||"Slidev");R({title:`Presenter - ${g}`}),y(!1);const{timer:x,resetTimer:c}=U(),u=y([]),n=$(()=>M.value<E.value?{route:k.value,clicks:M.value+1}:j.value?{route:O.value,clicks:0}:null);return W(),Z(()=>{const z=r.value.querySelector("#slide-content"),s=G(X()),w=J();K(()=>{if(!w.value||Y.value||!ee.value)return;const d=z.getBoundingClientRect(),p=(s.x-d.left)/d.width*100,m=(s.y-d.top)/d.height*100;if(!(p<0||p>100||m<0||m>100))return{x:p,y:m}},d=>{Q.cursor=d})}),(z,s)=>{const w=Ie,d=Te,p=Me,m=be;return i(),_(re,null,[e("div",je,[e("div",{class:C(["grid-container",`layout${t(te)}`])},[e("div",Oe,[We,Ze,e("div",{class:"timer-btn my-auto relative w-22px h-22px cursor-pointer text-lg",opacity:"50 hover:100",onClick:s[0]||(s[0]=(...a)=>t(c)&&t(c)(...a))},[o(w,{class:"absolute"}),o(d,{class:"absolute opacity-0"})]),e("div",Ge,se(t(x)),1)]),e("div",{ref_key:"main",ref:r,class:"relative grid-section main flex flex-col p-2 lg:p-4",style:v(t(P))},[o(B,{key:"main",class:"h-full w-full"},{default:V(()=>[o(ce,{"render-context":"presenter"})]),_:1}),Xe],4),e("div",{class:"relative grid-section next flex flex-col p-2 lg:p-4",style:v(t(P))},[n.value?(i(),h(B,{key:"next",class:"h-full w-full"},{default:V(()=>{var a;return[o(t(de),{is:(a=n.value.route)==null?void 0:a.component,"clicks-elements":u.value,"onUpdate:clicksElements":s[1]||(s[1]=D=>u.value=D),clicks:n.value.clicks,"clicks-disabled":!1,class:C(t(ue)(n.value.route)),route:n.value.route,"render-context":"previewNext"},null,8,["is","clicks-elements","clicks","class","route"])]}),_:1})):S("v-if",!0),Je],4),S(" Notes "),(i(),_("div",Ke,[(i(),h(Ee,{key:1,class:"w-full max-w-full h-full overflow-auto p-2 lg:p-4",style:v({fontSize:`${t(oe)}em`})},null,8,["style"])),e("div",Qe,[e("button",{class:"slidev-icon-btn",onClick:s[2]||(s[2]=(...a)=>t(H)&&t(H)(...a))},[o(L,{text:"Increase font size"}),o(p)]),e("button",{class:"slidev-icon-btn",onClick:s[3]||(s[3]=(...a)=>t(T)&&t(T)(...a))},[o(L,{text:"Decrease font size"}),o(m)]),S("v-if",!0)])])),e("div",Ye,[o(_e,{persist:!0})]),(i(),h(fe,{key:2}))],2),e("div",et,[e("div",{class:"progress h-2px bg-primary transition-all",style:v({width:`${(t(ne)-1)/(t(le)-1)*100}%`})},null,4)])]),o(pe),o(ie,{modelValue:t(b),"onUpdate:modelValue":s[5]||(s[5]=a=>ae(b)?b.value=a:null)},null,8,["modelValue"])],64)}}}),lt=q(tt,[["__scopeId","data-v-574fd206"],["__file","/Users/martin/Projects/pugs/presentations/logseq-faiss-chatgpt/slidev/node_modules/@slidev/client/internals/Presenter.vue"]]);export{lt as default};