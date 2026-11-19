/* Offline spatial labels. No API calls, storage services or inferred labels. */
(function () {
'use strict';
const schema='pixel-whitening-labels-v1', corners=['TL','TR','BL','BR'];
const allowed=['whitening','not_whitening','uncertain'];
const colors={whitening:'#00a040',not_whitening:'#e02030',uncertain:'#e89b00'};
const deep=x=>JSON.parse(JSON.stringify(x));
const embedded=document.getElementById('pixel-annotation-state');
const initial=JSON.parse(embedded.textContent);
let state=deep(initial);
const editors=new Map();
const globalStatus=document.getElementById('pixel-label-status');
const safeJson=x=>JSON.stringify(x).replace(/</g,'\\u003c');
function tileSize(card){return (card.result.measurement_image_size[0]-10)/2;}
function hasLabels(card){return card.annotations.length>0 || card.complete_corners.length>0;}
function matchesPoint(box,x,y,pad=0){return x>=box[0]-pad && y>=box[1]-pad && x<=box[0]+box[2]+pad && y<=box[1]+box[3]+pad;}
function rectangle(a,b,size){const x=Math.max(0,Math.min(a.x,b.x)),y=Math.max(0,Math.min(a.y,b.y));return [x,y,Math.min(size,Math.max(a.x,b.x))-x,Math.min(size,Math.max(a.y,b.y))-y];}
function annotationBox(c){return c.box;}
function candidates(card,corner){
 const decisions=new Map((card.result.whitening_classification?.decisions||[]).map(d=>[d.id,d]));
 return [...card.result.candidates,...card.result.excluded_candidates].filter(c=>c.corner===corner).map(c=>{
  const d=decisions.get(c.id);let box=c.box_in_corner;
  if(d?.reason==='compact_light_patch_supported_at_two_scales'){
   const patches=c.shape?.patch_evidence?.scales||[];
   if(patches.length)box=patches.reduce((a,b)=>a.scale>b.scale?a:b).box_in_corner;
  }
  return {id:c.id,corner,box};
 });
}
function captureNotes(){for(const c of state.cards){const note=document.getElementById('pixel-note-'+c.number);if(note)c.notes=note.value;}}
function sync(){captureNotes();embedded.textContent=safeJson(state);}
function changed(){sync();globalStatus.textContent='Unsaved labels — save the marked review or export the labeled dataset.';}
function snapshot(card){return {annotations:deep(card.annotations),complete_corners:[...card.complete_corners]};}
function remember(editor){editor.history.push(snapshot(editor.card));if(editor.history.length>100)editor.history.shift();}
function relatedCards(card){
 const found=new Set([card]);let previous=0;
 while(previous!==found.size){previous=found.size;for(const c of state.cards){if([...found].some(a=>a.group_id===c.group_id||a.listing===c.listing||a.source_sha256===c.source_sha256))found.add(c);}}
 return [...found];
}
function groupLocked(card){return relatedCards(card).some(c=>hasLabels(c)||c.previously_reviewed||c.split_locked||c.notes);}
function lockGroup(card){for(const c of relatedCards(card))c.split_locked=true;}
function groupHash(text){let h=2166136261;for(const c of text){h^=c.charCodeAt(0);h=Math.imul(h,16777619);}return h>>>0;}
function candidatesAt(card,corner,x,y){return candidates(card,corner).filter(c=>matchesPoint(c.box,x,y,4)).sort((a,b)=>a.box[2]*a.box[3]-b.box[2]*b.box[3]);}
function addMark(editor,box,candidateId){
 const c=editor.card,corner=editor.tools.corner.value,label=editor.tools.label.value;
 remember(editor);
 lockGroup(c);
 const old=candidateId===undefined?null:c.annotations.find(a=>a.corner===corner&&a.candidate_id===candidateId);
 if(old){old.label=label;editor.selected=old.id;}
 else {const id='mark-'+Date.now().toString(36)+'-'+Math.random().toString(36).slice(2,9);
  const rounded=box.map(x=>Math.round(x*100)/100),size=tileSize(c);
  rounded[2]=Math.min(rounded[2],size-rounded[0]);rounded[3]=Math.min(rounded[3],size-rounded[1]);
  const mark={id,corner,label,box:rounded};
  if(candidateId!==undefined)mark.candidate_id=candidateId;
  c.annotations.push(mark);editor.selected=id;
 }
 changed();refreshAll();
}
function draw(editor){
 const {card,tools,image}=editor,size=tileSize(card),corner=tools.corner.value,ctx=tools.canvas.getContext('2d');
 if(!image.complete)return;
 const x=corner[1]==='R'?size+10:0,y=corner[0]==='B'?size+50:20;
 ctx.clearRect(0,0,size,size);ctx.drawImage(image,x,y,size,size,0,0,size,size);
 function box(b,color,text,width=1){ctx.strokeStyle='#000';ctx.lineWidth=width+2;ctx.strokeRect(...b);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.strokeRect(...b);ctx.font='13px sans-serif';ctx.lineWidth=3;ctx.strokeText(text,b[0],Math.max(14,b[1]-4));ctx.fillStyle=color;ctx.fillText(text,b[0],Math.max(14,b[1]-4));}
 if(tools['show-candidates'].checked){for(const c of candidates(card,corner))box(c.box,'#55baff',String(c.id));}
 for(const a of card.annotations.filter(a=>a.corner===corner))box(annotationBox(a),colors[a.label],a.label.replace('_',' ')+(a.candidate_id!==undefined?' #'+a.candidate_id:''),a.id===editor.selected?3:2);
 if(editor.drag?.end)box(rectangle(editor.drag.start,editor.drag.end,size),colors[tools.label.value],'new mark',2);
}
function refresh(editor){
 const {card,tools}=editor,size=tileSize(card),zoom=Number(tools.zoom.value);
 tools.canvas.width=size;tools.canvas.height=size;tools.canvas.style.width=(size*zoom)+'px';tools.canvas.style.height=(size*zoom)+'px';
 tools.group.value=card.group_id;tools.split.value=card.split;
 tools.group.disabled=groupLocked(card);tools.split.disabled=groupLocked(card);
 tools.status.textContent=(card.split==='validation'?'Reserved validation':'Development')+' · '+card.annotations.length+' labels · '+card.complete_corners.length+'/4 fully reviewed corners'+(card.previously_reviewed?' · Previously reviewed: development only.':'')+(card.result.analysis_mode!=='whitening_classification'?' · Marking is available, but tuning needs a run made with edge_classifier.':'');
 tools.marks.replaceChildren();
 for(const a of card.annotations.filter(a=>a.corner===tools.corner.value)){
  const li=document.createElement('li');li.textContent=a.label.replace('_',' ')+(a.candidate_id!==undefined?' — candidate '+a.candidate_id:' — drawn region');
  const b=document.createElement('button');b.type='button';b.textContent='Select';b.addEventListener('click',()=>{editor.selected=a.id;draw(editor);});li.append(b);tools.marks.append(li);
 }
 draw(editor);
}
function refreshAll(){for(const e of editors.values())refresh(e);}
function validate(payload){
 if(!payload||payload.schema!==schema||!Array.isArray(payload.cards))throw Error('This is not a labeled pixel-review dataset.');
 if(payload.cards.length!==initial.cards.length)throw Error('Import must contain every card from this review.');
 const seen=new Set(),groups=new Map();
 const validated=payload.cards.map(c=>{
  const orig=initial.cards.find(x=>x.number===c.number);
  if(!orig||seen.has(c.number)||c.listing!==orig.listing||c.source_sha256!==orig.source_sha256)throw Error('The imported labels belong to different photos or duplicate cards.');
  seen.add(c.number);
  if(typeof c.group_id!=='string'||!c.group_id.trim()||c.group_id.length>200||!['development','validation'].includes(c.split))throw Error('Invalid card group or dataset split.');
  if(orig.previously_reviewed&&c.split!=='development')throw Error('Previously reviewed cards must remain development cards.');
  const current=state.cards.find(x=>x.number===c.number);
  if(current.previously_reviewed&&c.split!=='development')throw Error('Retired validation cards must remain development cards.');
  if(groupLocked(current)&&(c.split!==current.split||c.group_id!==current.group_id))throw Error('A reviewed card cannot be reassigned through import. Use the retire button to move reserved cards to development.');
  for(const key of ['group:'+c.group_id,'listing:'+c.listing,'photo:'+c.source_sha256]){
   if(groups.has(key)&&groups.get(key)!==c.split)throw Error('Photos of one physical card cannot be split across development and validation.');
   groups.set(key,c.split);
  }
  if(typeof c.notes!=='string'||!Array.isArray(c.annotations)||c.annotations.length>10000||!Array.isArray(c.complete_corners)||new Set(c.complete_corners).size!==c.complete_corners.length||c.complete_corners.some(x=>!corners.includes(x)))throw Error('Invalid labels, notes or completed corners.');
  const ids=new Set(),size=tileSize(orig);
  const annotations=c.annotations.map(a=>{
   if(!a||typeof a.id!=='string'||!a.id||ids.has(a.id)||!corners.includes(a.corner)||!allowed.includes(a.label)||!Array.isArray(a.box)||a.box.length!==4||a.box.some(x=>!Number.isFinite(x)))throw Error('Invalid annotation.');
   const [x,y,w,h]=a.box;if(x<0||y<0||w<=0||h<=0||x+w>size+.01||y+h>size+.01)throw Error('Annotation is outside the corner image.');
   ids.add(a.id);const clean={id:a.id,corner:a.corner,label:a.label,box:[...a.box]};
   if(a.candidate_id!==undefined){if(!candidates(orig,a.corner).some(x=>x.id===a.candidate_id))throw Error('Unknown candidate ID.');clean.candidate_id=a.candidate_id;}
   return clean;
  });
  return {...deep(orig),group_id:c.group_id,split:c.split,notes:c.notes,complete_corners:[...c.complete_corners],annotations,split_locked:Boolean(c.split_locked||hasLabels(c))};
 });
 return {schema,cards:validated};
}
function download(content,type,name){const url=URL.createObjectURL(new Blob([content],{type}));const a=document.createElement('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),10000);globalStatus.textContent='Download requested. Keep the downloaded file before closing.';}
async function exportData(){
 sync();const exported=deep(state);
 for(const c of exported.cards)c.positive_only_review=Boolean(c.annotations.length||c.notes.trim());
 const text=JSON.stringify({...exported,exported_at:new Date().toISOString()});
 try{
  if(typeof CompressionStream!=='undefined'){
   const stream=new Blob([text]).stream().pipeThrough(new CompressionStream('gzip'));
   const packed=await new Response(stream).arrayBuffer();
   download(packed,'application/gzip','pixel-whitening-labels.json.gz');return;
  }
 }catch(error){globalStatus.textContent='Compression unavailable; exporting ordinary JSON.';}
 download(text,'application/json','pixel-whitening-labels.json');
}
function saveHtml(){
 sync();const copy=document.documentElement.cloneNode(true);
 for(const c of state.cards){const n=copy.querySelector('#pixel-note-'+c.number);if(n)n.textContent=c.notes;}
 copy.querySelector('#pixel-annotation-state').textContent=safeJson(state);
 copy.querySelector('#pixel-label-status').textContent='Saved markings loaded. Save again after editing.';
 download('<!doctype html>\n'+copy.outerHTML,'text/html;charset=utf-8','reviewed-pixel-markings.html');
}
// Validate embedded saved state too. Result snapshots remain the original report's data.
state=validate(state);
for(const card of state.cards){
 const element=document.getElementById('pixel-editor-'+card.number);if(!element)continue;
 const tools={};element.querySelectorAll('[data-tool]').forEach(e=>tools[e.dataset.tool]=e);
 const sheet=element.closest('section').querySelector('.comparison figure img');
 const image=new Image();const editor={card,tools,image,history:[],selected:null,drag:null};editors.set(card.number,editor);
 image.onload=()=>refresh(editor);image.src=sheet.src;
 for(const key of ['corner','zoom','show-candidates'])tools[key].addEventListener('change',()=>{editor.selected=null;editor.drag=null;refresh(editor);});
 tools.group.addEventListener('change',()=>{const value=tools.group.value.trim();const other=state.cards.find(c=>c!==card&&c.group_id===value);if(!value||value.length>200||(other&&other.split!==card.split)){tools.status.textContent='Choose a nonempty group. Existing photos in that group must use the same split.';tools.group.value=card.group_id;return;}card.group_id=value;changed();refreshAll();});
 tools.split.addEventListener('change',()=>{if(groupLocked(card)){refresh(editor);return;}for(const c of relatedCards(card))c.split=tools.split.value;changed();refreshAll();});
 tools.undo.addEventListener('click',()=>{const old=editor.history.pop();if(old){Object.assign(card,old);editor.selected=null;changed();refreshAll();}});
 tools.delete.addEventListener('click',()=>{if(editor.selected){remember(editor);card.annotations=card.annotations.filter(a=>a.id!==editor.selected);editor.selected=null;changed();refreshAll();}});
 const point=e=>{const rect=tools.canvas.getBoundingClientRect(),size=tileSize(card);return{x:Math.max(0,Math.min(size,(e.clientX-rect.left)*size/rect.width)),y:Math.max(0,Math.min(size,(e.clientY-rect.top)*size/rect.height))};};
 tools.canvas.addEventListener('pointerdown',e=>{if(e.button!==undefined&&e.button!==0)return;e.preventDefault();editor.drag={start:point(e),end:null};tools.canvas.setPointerCapture?.(e.pointerId);});
 tools.canvas.addEventListener('pointermove',e=>{if(editor.drag){editor.drag.end=point(e);draw(editor);}});
 tools.canvas.addEventListener('pointercancel',()=>{editor.drag=null;draw(editor);});
 tools.canvas.addEventListener('pointerup',e=>{
  if(!editor.drag)return;const a=editor.drag.start,b=point(e),box=rectangle(a,b,tileSize(card));editor.drag=null;
  if(box[2]>=2&&box[3]>=2){addMark(editor,box);return;}
  const current=card.annotations.filter(m=>m.corner===tools.corner.value&&matchesPoint(m.box,b.x,b.y,2)).reverse()[0];
  if(current){editor.selected=current.id;if(current.label!==tools.label.value){remember(editor);current.label=tools.label.value;changed();}refreshAll();return;}
  const hit=tools['show-candidates'].checked?candidatesAt(card,tools.corner.value,b.x,b.y)[0]:null;
  if(hit)addMark(editor,hit.box,hit.id);else draw(editor);
 });
 const note=document.getElementById('pixel-note-'+card.number);if(note){note.value=card.notes;note.addEventListener('input',()=>{lockGroup(card);changed();refreshAll();});}
 refresh(editor);
}
document.getElementById('pixel-label-save').addEventListener('click',saveHtml);
document.getElementById('pixel-label-export').addEventListener('click',exportData);
document.getElementById('pixel-label-retire').addEventListener('click',()=>{
 if(!confirm('Retire this review’s reserved cards to development? Use this after validation results have guided changes. These cards cannot be reserved again.'))return;
 for(const c of state.cards){if(c.split==='validation'){c.split='development';c.previously_reviewed=true;c.split_locked=true;}}
 changed();refreshAll();
});
document.getElementById('pixel-label-reserve').addEventListener('click',()=>{
 captureNotes();const processed=new Set();let reserved=0,eligible=0;
 for(const c of state.cards){if(processed.has(c.number))continue;const group=relatedCards(c);group.forEach(x=>processed.add(x.number));if(groupLocked(c))continue;
  const key=group.map(x=>x.group_id).sort()[0];const split=groupHash(key)%10<3?'validation':'development';
  for(const x of group)x.split=split;eligible++;if(split==='validation')reserved++;
 }
 changed();refreshAll();globalStatus.textContent='Reserved '+reserved+' of '+eligible+' eligible groups. Small batches may differ from 30%. Previously reviewed groups were left alone. Save this assignment before labeling.';
});
document.getElementById('pixel-label-import').addEventListener('change',async e=>{
 try{const file=e.target.files[0];if(!file)return;const incoming=validate(JSON.parse(file.name?.endsWith('.gz')?await new Response(file.stream().pipeThrough(new DecompressionStream('gzip'))).text():await file.text()));
  if(!confirm('Replace this page’s current markings with the imported labels? Save current work first if needed.'))return;
  // Keep existing objects: canvas handlers retain their card references.
  for(const card of state.cards){const imported=incoming.cards.find(c=>c.number===card.number);Object.assign(card,imported);const ed=editors.get(card.number);if(ed){ed.history=[];ed.selected=null;}const n=document.getElementById('pixel-note-'+card.number);if(n)n.value=card.notes;}
  sync();refreshAll();globalStatus.textContent='Imported labels. Save the marked review to keep them.';
 }catch(error){globalStatus.textContent='Import failed: '+error.message;}
});
document.getElementById('pixel-save-html')?.addEventListener('click',e=>{e.preventDefault();e.stopImmediatePropagation();saveHtml();},true);
window.pixelAnnotationTools={validate,rectangle,candidatesAt,snapshot,exportData,saveHtml,getState:()=>deep(state)};
})();
