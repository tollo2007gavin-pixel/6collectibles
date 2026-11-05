// Offline interaction tests against the actual embedded script; no dependencies.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const template=fs.readFileSync(__dirname+'/card_review.html','utf8');
const script=template.match(/<script type="text\/javascript">([\s\S]*?)<\/script>/)[1];
const batch={schema_version:1,batch_id:'test',batch_sha256:'hash',portable:true,pipeline:'api',cards:Array.from({length:10},(_,i)=>({id:String(i),number:i+1,title:'Card '+i,url:'',pipeline_decision:{status:'pass',whitening:'clean',reason:'clean_back',evidence:[]},images:[],timings:{}}))};
const storage=new Map();
function boot(embedded=null,failStorage=false){
 const elements={},downloads=[];
 function element(){return {value:'',options:[],files:[],textContent:'',classList:{toggle(){}},addEventListener(t,f){this[t]=f},append(...e){this.options.push(...e)},replaceChildren(...e){this.options=e},click(){if(this.onclick)this.onclick()}}}
 const get=id=>elements[id]||(elements[id]=element());get('batch-data').textContent=JSON.stringify(batch);get('saved-review').textContent=JSON.stringify(embedded);
 const ctx={document:{getElementById:get,createElement:element,documentElement:{cloneNode(){const saved=element();return {querySelector(id){return id==='#saved-review'?saved:element()},get outerHTML(){return template.replace('__BATCH_DATA__',JSON.stringify(batch)).replace('<script id="saved-review" type="application/json">null</script>','<script id="saved-review" type="application/json">'+saved.textContent+'</script>')}}}}},localStorage:{getItem:k=>{if(failStorage)throw Error();return storage.get(k)},setItem:(k,v)=>{if(failStorage)throw Error();storage.set(k,v)}},Blob,URL:{createObjectURL:b=>{downloads.push(b);return 'blob:test'},revokeObjectURL(){}},Date,setTimeout:f=>f(),confirm:()=>true};vm.createContext(ctx);vm.runInContext(script,ctx);return {get,downloads,run:s=>vm.runInContext(s,ctx)};
}
(async()=>{
 let t=boot();assert.equal(t.run('payload().summary.unreviewed'),10);
 t.get('assessment').value='incorrect';t.get('correction').value='not_clean';t.get('notes').value='Chip </script> on corner';t.run('capture()');t.get('next').onclick();t.get('assessment').value='correct';t.run('capture()');
 t.get('export').onclick();const exported=JSON.parse(await t.downloads.at(-1).text());assert.equal(exported.summary.incorrect,1);assert.equal(exported.summary.correct,1);assert.equal(exported.cards[0].pipeline_decision.whitening,'clean');
 t.get('saveHtml').onclick();const savedHtml=await t.downloads.at(-1).text();const saved=JSON.parse(savedHtml.match(/<script id="saved-review" type="application\/json">(.*?)<\/script>/s)[1]);assert.equal(saved.cards[0].human_review.notes,'Chip </script> on corner');
 storage.clear();t=boot(saved);assert.equal(t.run('payload().summary.correct'),1);assert.equal(t.run('payload().summary.incorrect'),1);t.get('unreviewed').onclick();assert.equal(t.run('index'),2);
 t=boot(null,true);assert.equal(t.run('payload().summary.unreviewed'),10);t.get('importFile').files=[{text:async()=>JSON.stringify(exported)}];await t.get('importFile').onchange();assert.equal(t.run('payload().summary.incorrect'),1);
 for(const mutate of [p=>p.batch_sha256='wrong',p=>p.cards.pop(),p=>p.cards[1].id='0',p=>p.cards[0].human_review.assessment='bad']){let p=structuredClone(exported);mutate(p);assert.throws(()=>t.run('validate('+JSON.stringify(p)+')'))}
 console.log('UI passed: blank answers, edits, navigation, export JSON, saved HTML restore, blocked storage, import and malformed/cross-batch rejection.');
})().catch(e=>{console.error(e);process.exitCode=1});
