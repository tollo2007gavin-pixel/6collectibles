/* Actual annotation script in a minimal offline DOM; no npm dependencies. */
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const script=fs.readFileSync(__dirname+'/pixel_annotations.js','utf8');
const base={schema:'pixel-whitening-labels-v1',cards:[1,2].map(number=>({number,listing:'listing-'+number,source_sha256:'hash-'+number,group_id:'listing-'+number,split:'development',notes:'',complete_corners:[],annotations:[],previously_reviewed:false,result:{analysis_mode:'whitening_classification',measurement_image_size:[1010,1050],candidates:[{id:1,corner:'TL',box_in_corner:[100,100,10,10]}],excluded_candidates:[],whitening_classification:{decisions:[{id:1,classification:'likely_whitening',reason:'strong_card_associated_edge_patch'}]}}}))};
function boot(input=base,compression=false){
 const elements={},downloads=[];
 function element(){return {value:'',textContent:'',checked:false,disabled:false,style:{},dataset:{},children:[],handlers:{},files:[],
  addEventListener(name,fn){(this.handlers[name]??=[]).push(fn)},
  async fire(name,event={}){for(const fn of this.handlers[name]||[])await fn({target:this,preventDefault(){},stopImmediatePropagation(){},...event})},
  append(...items){this.children.push(...items)},replaceChildren(...items){this.children=items},remove(){},click(){},
  getContext(){return {clearRect(){},drawImage(){},strokeRect(){},strokeText(){},fillText(){}}},
  getBoundingClientRect(){return {left:0,top:0,width:parseFloat(this.style.width)||500,height:parseFloat(this.style.height)||500}},setPointerCapture(){}};}
 const get=id=>elements[id]||(elements[id]=element());
 get('pixel-annotation-state').textContent=JSON.stringify(input);
 for(const card of input.cards){
  const editor=get('pixel-editor-'+card.number);editor.tools={};
  for(const name of ['corner','label','zoom','show-candidates','canvas','group','split','status','marks','undo','delete']){const e=element();e.dataset.tool=name;editor.tools[name]=e;}
  Object.assign(editor.tools.corner,{value:'TL'});editor.tools.label.value='whitening';editor.tools.zoom.value='1';editor.tools['show-candidates'].checked=true;
  editor.querySelectorAll=()=>Object.values(editor.tools);
  editor.closest=()=>({querySelector:()=>({src:'data:image/png;base64,AA=='})});
 }
 const ctx={document:{getElementById:get,createElement:element,body:element(),documentElement:{cloneNode(){const saved={};return {querySelector(selector){return saved[selector]??=(Object.assign(element(),{textContent:get(selector.slice(1)).textContent}))},get outerHTML(){return '<html><script id="pixel-annotation-state" type="application/json">'+saved['#pixel-annotation-state'].textContent+'</script></html>';}}}}},Image:class{constructor(){this.complete=false}set src(v){this.complete=true;this.onload?.()}},Blob,URL:{createObjectURL:b=>{downloads.push(b);return 'blob:test'},revokeObjectURL(){}},setTimeout:f=>f(),confirm:()=>true,Date,Math,window:{}};
 if(compression)Object.assign(ctx,{CompressionStream,DecompressionStream,Response});
 vm.createContext(ctx);vm.runInContext(script,ctx);
 return {get,downloads,api:ctx.window.pixelAnnotationTools,tools:n=>get('pixel-editor-'+n).tools};
}
(async()=>{
 let t=boot(),tools=t.tools(1);
 // Reserve before marking; all subsequent label edits retain the split.
 tools.split.value='validation';await tools.split.fire('change');
 await tools.canvas.fire('pointerdown',{clientX:105,clientY:105,button:0});await tools.canvas.fire('pointerup',{clientX:105,clientY:105});
 let c=t.api.getState().cards[0];assert.equal(c.annotations[0].candidate_id,1);assert.equal(c.annotations[0].label,'whitening');assert.equal(c.split,'validation');assert(tools.split.disabled);
 // Undo does not make a previously exposed card eligible for moving splits.
 await tools.undo.fire('click');assert.equal(t.api.getState().cards[0].annotations.length,0);assert(tools.split.disabled);
 // Zoomed rectangle converts screen coordinates back to exact saved image pixels.
 tools.zoom.value='2';await tools.zoom.fire('change');tools.label.value='not_whitening';
 await tools.canvas.fire('pointerdown',{clientX:400,clientY:400,button:0});await tools.canvas.fire('pointermove',{clientX:440,clientY:460});await tools.canvas.fire('pointerup',{clientX:440,clientY:460});
 c=t.api.getState().cards[0];assert.deepEqual(Array.from(c.annotations[0].box),[200,200,20,30]);assert.equal(c.annotations[0].label,'not_whitening');assert.deepEqual(Array.from(c.complete_corners),[]);
 t.get('pixel-note-1').value='Chip </script> and </textarea>';await t.get('pixel-note-1').fire('input');
 await t.get('pixel-label-export').fire('click');const exported=JSON.parse(await t.downloads.at(-1).text());assert.equal(exported.cards[0].notes,'Chip </script> and </textarea>');
 await t.get('pixel-label-save').fire('click');const html=await t.downloads.at(-1).text();const restored=JSON.parse(html.match(/<script id="pixel-annotation-state" type="application\/json">(.*?)<\/script>/s)[1]);
 assert.equal(restored.cards[0].annotations.length,1);assert.equal(restored.cards[0].notes,'Chip </script> and </textarea>');
 t=boot(restored);assert.equal(t.api.getState().cards[0].annotations.length,1);
 // Import reuses live card objects: further canvas edits must affect exported state.
 t=boot();t.get('pixel-label-import').files=[{text:async()=>JSON.stringify(exported)}];await t.get('pixel-label-import').fire('change');
 assert.equal(t.api.getState().cards[0].annotations.length,1);tools=t.tools(1);tools.label.value='uncertain';
 await tools.canvas.fire('pointerdown',{clientX:300,clientY:300,button:0});await tools.canvas.fire('pointerup',{clientX:320,clientY:320});
 assert.equal(t.api.getState().cards[0].annotations.length,2);assert.equal(t.api.getState().cards[0].annotations[1].label,'uncertain');
 // Malformed or other-photo labels never silently attach to this batch.
 for(const mutate of [p=>p.schema='bad',p=>p.cards.pop(),p=>p.cards[0].source_sha256='wrong',p=>p.cards[0].annotations[0].box=[490,490,30,30],p=>p.cards[0].annotations[0].label='clean',p=>p.cards[1].group_id=p.cards[0].group_id]){
  const p=structuredClone(exported);mutate(p);assert.throws(()=>t.api.validate(p));
 }
 // Cached measurements cannot be replaced by imported alleged predictions.
 const altered=structuredClone(exported);altered.cards[0].result.candidates=[];assert.equal(t.api.validate(altered).cards[0].result.candidates.length,1);
 const reviewed=structuredClone(base);reviewed.cards[0].previously_reviewed=true;reviewed.cards[0].notes='Already discussed';t=boot(reviewed);
 const bad=structuredClone(reviewed);bad.cards[0].split='validation';assert.throws(()=>t.api.validate(bad));
 await t.get('pixel-label-reserve').fire('click');assert.equal(t.api.getState().cards[0].split,'development');
 // Same listing is linked even if manually assigned a different group string.
 const duplicate=structuredClone(base);duplicate.cards[1].listing=duplicate.cards[0].listing;t=boot(duplicate);tools=t.tools(1);tools.split.value='validation';await tools.split.fire('change');assert(t.api.getState().cards.every(c=>c.split==='validation'));
 await t.get('pixel-label-retire').fire('click');assert(t.api.getState().cards.every(c=>c.split==='development'&&c.previously_reviewed));
 const retired=t.api.getState();retired.cards[0].split='validation';assert.throws(()=>t.api.validate(retired));
 const compressed=boot(base,true);await compressed.get('pixel-label-export').fire('click');const zipped=Buffer.from(await compressed.downloads.at(-1).arrayBuffer());assert.equal(JSON.parse(require('zlib').gunzipSync(zipped).toString()).schema,base.schema);
 console.log('Annotation UI passed: candidate clicks, zoomed coordinates, labels, completeness, immutable split after review, notes/HTML round trip, import/edit/export, malformed labels and same-card grouping.');
})().catch(e=>{console.error(e);process.exitCode=1});
