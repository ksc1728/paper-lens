import { useEffect, useRef, useState } from 'react'
import { BookOpen, FileText, Search, Sparkles, Trash2, UploadCloud } from 'lucide-react'

const api = async (path, options) => {
  const response = await fetch(path, options)
  const data = await response.json().catch(() => ({}))
  if (!response.ok) throw new Error(data.detail || 'Something went wrong')
  return data
}

export default function App() {
  const [documents, setDocuments] = useState([])
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const inputRef = useRef(null)

  const refresh = () => api('/api/documents').then(d => setDocuments(d.documents)).catch(e => setError(e.message))
  useEffect(() => { refresh() }, [])

  const upload = async files => {
    if (!files.length) return
    setBusy('upload'); setError('')
    const body = new FormData(); body.append('files', files[0])
    try { await api('/api/documents', { method: 'POST', body }); await refresh() }
    catch (e) { setError(e.message) }
    finally { setBusy('') }
  }

  const ask = async e => {
    e.preventDefault(); if (!question.trim()) return
    setBusy('ask'); setError(''); setResult(null)
    try { setResult(await api('/api/ask', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({question}) })) }
    catch (e) { setError(e.message) }
    finally { setBusy('') }
  }

  const clear = async () => {
    if (!confirm('Remove the uploaded paper and its FAISS index?')) return
    await api('/api/documents', {method:'DELETE'}); setDocuments([]); setResult(null)
  }

  return <main>
    <header><div className="brand"><span><BookOpen size={22}/></span> PaperLens</div><p>Research answers, traced to the page.</p></header>
    <section className="hero">
      <div className="eyebrow"><Sparkles size={14}/> DOCUMENT-GROUNDED RESEARCH</div>
      <h1>Understand one paper.</h1><h2><em>Keep the evidence visible.</em></h2>
      <p>Upload a research PDF, inspect its structure, ask questions, and trace every cited passage back to the page.</p>
    </section>

    {error && <div className="error">{error}</div>}

    <div className="workspace">
      <aside>
        <div className="section-title"><span>Uploaded paper</span>{documents.length > 0 && <button className="icon" onClick={clear} aria-label="Remove paper"><Trash2 size={16}/></button>}</div>
        <button className="dropzone" onClick={() => inputRef.current.click()} onDrop={e => {e.preventDefault(); upload(e.dataTransfer.files)}} onDragOver={e => e.preventDefault()}>
          <UploadCloud size={26}/><strong>{busy === 'upload' ? 'Detecting sections…' : documents.length ? 'Replace research PDF' : 'Upload research PDF'}</strong><small>One text-based PDF · up to 25 MB</small>
        </button>
        <input ref={inputRef} hidden type="file" accept="application/pdf" onChange={e => upload(e.target.files)}/>
        <div className="papers">
          {documents.map(doc => <div className="paper" key={doc.name}><FileText size={18}/><div><strong>{doc.title || doc.name}</strong><small>{doc.pages} pages · {doc.sections} sections · {doc.chunks} passages</small></div></div>)}
          {!documents.length && <p className="empty">No paper indexed yet.</p>}
        </div>
      </aside>

      <section className="qa">
        <form onSubmit={ask}>
          <label htmlFor="question">Ask about the paper</label>
          <div className="searchbox"><Search size={20}/><input id="question" value={question} onChange={e => setQuestion(e.target.value)} placeholder="Summarize the Methods section"/><button disabled={!documents.length || busy === 'ask'}>{busy === 'ask' ? 'Reading…' : 'Ask'}</button></div>
          {!documents.length && <small>Upload one paper to begin.</small>}
        </form>

        {!result && <div className="blank"><div><Sparkles size={24}/></div><h2>A grounded answer will appear here</h2><p>Each answer includes the source paper, page number, similarity score, and retrieved text.</p></div>}
        {result && <article className="answer">
          <div className="answer-head"><span>ANSWER</span><small>{result.mode === 'extractive' ? 'Retrieval-only mode' : result.mode === 'metadata' ? 'Verified PDF metadata' : result.mode === 'abstract' ? 'Extracted abstract' : `Generated with ${result.mode}`}</small></div>
          <p className="answer-text">{result.answer.split(/(\[\d+\])/g).map((part, index) => {
            const match = part.match(/^\[(\d+)\]$/)
            return match ? <a className="citation-link" href={`#citation-${match[1]}`} key={index}>{part}</a> : part
          })}</p>
          {!!result.sources.length && <h3>Citations</h3>}
          <div className="sources">{result.sources.map((source, i) => <details id={`citation-${i+1}`} key={`${source.paper}-${source.page}-${i}`} open={i === 0}>
            <summary><span className="number">{i+1}</span><span><strong>Citation [{i+1}] — {source.paper} — Page {source.page}</strong><small>{source.section_name}{source.score < 0.999 ? ` · similarity ${source.score.toFixed(3)}` : ''}</small></span></summary>
            <p>{source.text}</p>
          </details>)}</div>
        </article>}
      </section>
    </div>
    <footer>Section-aware retrieval with Sentence Transformers + FAISS · Page-level citations</footer>
  </main>
}
