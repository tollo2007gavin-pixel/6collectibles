// Offline interaction tests against the actual embedded script; no dependencies.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const template=fs.readFileSync(__dirname+'/card_review.html','utf8');
const script=template.match(/<script type="text\/javascript">([\s\S]*?)<\/script>/)[1];
const batch={schema_version:1,batch_id:'test',batch_sha256:'hash',portable:true,pipeline:'api',cards:Array.from({length:10},(_,i)=>({id:String(i),number:i+1,title:'Card '+i,url:'',pipeline_decision:{status:'pass',whitening:'clean',reason:'clean_back',evidence:[]},images:[{label:'Original listing photo',file:'original.png',sha256:'imagehash',src:'data:image/png;base64,AA=='}],timings:{}}))};
const storage=new Map();
function boot(embedded=null,failStorage=false){
 const elements={},downloads=[];
 function element(){return {getBoundingClientRect(){return {left:0,top:0,width:450,height:600}},getContext(){return {drawImage(){},beginPath(){},moveTo(){},lineTo(){},closePath(){},stroke(){},arc(){},fill(){},fillText(){}}},value:'',options:[],files:[],textContent:'',classList:{toggle(){}},addEventListener(t,f){this[t]=f},append(...e){this.options.push(...e)},replaceChildren(...e){this.options=e},click(){if(this.onclick)this.onclick()}}}
 const get=id=>elements[id]||(elements[id]=element());get('batch-data').textContent=JSON.stringify(batch);get('saved-review').textContent=JSON.stringify(embedded);
 const ctx={Image:class {constructor(){this.naturalWidth=450;this.naturalHeight=600}set src(v){this.onload()}},document:{getElementById:get,createElement:element,documentElement:{cloneNode(){const saved=element();return {querySelector(id){return id==='#saved-review'?saved:element()},get outerHTML(){return template.replace('__BATCH_DATA__',JSON.stringify(batch)).replace('<script id="saved-review" type="application/json">null</script>','<script id="saved-review" type="application/json">'+saved.textContent+'</script>')}}}}},localStorage:{getItem:k=>{if(failStorage)throw Error();return storage.get(k)},setItem:(k,v)=>{if(failStorage)throw Error();storage.set(k,v)}},Blob,URL:{createObjectURL:b=>{downloads.push(b);return 'blob:test'},revokeObjectURL(){}},Date,setTimeout:f=>f(),confirm:()=>true};vm.createContext(ctx);vm.runInContext(script,ctx);return {get,downloads,run:s=>vm.runInContext(s,ctx)};
}
(async()=>{
 let t=boot();assert.equal(t.run('payload().summary.unreviewed'),10);
 t.get('assessment').value='incorrect';t.get('correction').value='not_clean';t.get('notes').value='Chip </script> on corner';t.run('capture()');t.get('next').onclick();t.get('assessment').value='correct';t.run('capture()');
 t.get('export').onclick();const exported=JSON.parse(await t.downloads.at(-1).text());assert.equal(exported.summary.incorrect,1);assert.equal(exported.summary.correct,1);assert.equal(exported.cards[0].pipeline_decision.whitening,'clean');
 t.get('saveHtml').onclick();const savedHtml=await t.downloads.at(-1).text();const saved=JSON.parse(savedHtml.match(/<script id="saved-review" type="application\/json">(.*?)<\/script>/s)[1]);assert.equal(saved.cards[0].human_review.notes,'Chip </script> on corner');
 storage.clear();t=boot(saved);assert.equal(t.run('payload().summary.correct'),1);assert.equal(t.run('payload().summary.incorrect'),1);t.get('unreviewed').onclick();assert.equal(t.run('index'),2);
 t=boot(null,true);assert.equal(t.run('payload().summary.unreviewed'),10);t.get('importFile').files=[{text:async()=>JSON.stringify(exported)}];await t.get('importFile').onchange();assert.equal(t.run('payload().summary.incorrect'),1);
 for(const mutate of [p=>p.batch_sha256='wrong',p=>p.cards.pop(),p=>p.cards[1].id='0',p=>p.cards[0].human_review.assessment='bad']){let p=structuredClone(exported);mutate(p);assert.throws(()=>t.run('validate('+JSON.stringify(p)+')'))}
 t=boot(null,true);const corners={image_file:'original.png',image_sha256:'imagehash',width:450,height:600,points:[[.1,.1],[.9,.1],[.9,.9],[.1,.9]]};
 t.run('answers[cards[0].id]={assessment:"",corrected_decision:"",notes:"",manual_corners:validateManual(cards[0],'+JSON.stringify(corners)+')}');
 t.get('notes').value='Keep corrected points';t.run('capture()');t.get('saveHtml').onclick();const marked=await t.downloads.at(-1).text();const markedReview=JSON.parse(marked.match(/<script id="saved-review" type="application\/json">(.*?)<\/script>/s)[1]);
 storage.clear();t=boot(markedReview);assert.equal(t.run('payload().cards[0].human_review.manual_corners.image_sha256'),'imagehash');assert.equal(t.run('payload().cards[0].human_review.notes'),'Keep corrected points');
 for(const mutate of [p=>p.points.reverse(),p=>p.image_sha256='wrong',p=>p.points[0]=[2,0],p=>p.points=[[0,0],[1,1],[1,0],[0,1]]]){const bad=structuredClone(corners);mutate(bad);assert.throws(()=>t.run('validateManual(cards[0],'+JSON.stringify(bad)+')'))}
 t.get('clearCorners').onclick();assert.equal(t.run('payload().cards[0].human_review.manual_corners'),undefined);
 t.get('cornerPhoto').value='original.png';t.get('editCorners').onclick();
 for(const [x,y] of [[45,60],[405,60],[405,540],[45,540]])t.get('cornerCanvas').onclick({clientX:x,clientY:y});
 t.get('saveCorners').onclick();assert.equal(t.run('payload().cards[0].human_review.manual_corners.points.length'),4);assert.equal(t.run('payload().cards[0].pipeline_decision.whitening'),'clean');
 t.get('undoCorner').onclick();assert.equal(t.run('cornerPoints.length'),3);t.get('next').onclick();assert.equal(t.get('cornerCanvas').hidden,true);assert.equal(t.run('cornerPoints.length'),0);
 batch.pipeline='detection';storage.clear();t=boot(null,true);t.get('backSelection').value='missed_back';t.get('outlineQuality').value='cannot_assess';t.get('notes').value='Back is photo 3';t.run('capture()');
 assert.equal(t.run('payload().detection_summary.reviewed'),1);t.get('export').onclick();const detection=JSON.parse(await t.downloads.at(-1).text());
 t=boot(detection);assert.equal(t.get('backSelection').value,'missed_back');assert.equal(t.get('outlineQuality').value,'cannot_assess');t.get('unreviewed').onclick();assert.equal(t.run('index'),1);
 console.log('Detection feedback passed: separate answers, progress, export/restore and navigation.');
 console.log('Manual corners passed: preserve during notes edit, HTML restore, coordinate/photo validation, remove correction.');
 console.log('UI passed: blank answers, edits, navigation, export JSON, saved HTML restore, blocked storage, import and malformed/cross-batch rejection.');
})().catch(e=>{console.error(e);process.exitCode=1});
