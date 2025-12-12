'use client';

import React, { useState, useRef } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || '';

// === Types ===
type PreprocessingType = 'standard_scaler' | 'min_max_scaler' | 'drop_nulls';
type ModelType = 'logistic_regression' | 'decision_tree_classifier';
type StepType = PreprocessingType | 'train_test_split' | ModelType;

interface PipelineStep {
    id: string;
    type: StepType;
    params: Record<string, any>;
}

interface PipelineResult {
    success: boolean;
    message: string;
    metrics: Record<string, any>;
    logs: string[];
    model_path?: string;
}

interface DatasetMetadata {
    filename: string;
    rows: number;
    columns: number;
    column_names: string[];
    preview: Record<string, any>[];
    distributions?: { target_column: string; data: Record<string, number> };
    correlations?: { labels: string[]; data: number[][] };
    scatter_data?: { x: number; y: number; class: string }[];
}

// === Constants & Educational Content ===
const STEP_TEMPLATES: Record<StepType, { label: string; category: 'preprocessing' | 'model'; defaultParams: Record<string, any>; description: string }> = {
    standard_scaler: {
        label: 'Standard Scaler',
        category: 'preprocessing',
        defaultParams: {},
        description: "Standardizes features by removing the mean and scaling to unit variance. Useful when features have different units (e.g. Age vs Salary)."
    },
    min_max_scaler: {
        label: 'Min Max Scaler',
        category: 'preprocessing',
        defaultParams: { feature_range: '0,1' },
        description: "Scales features to a given range (usually 0 to 1). Keeps the relationship between values but fits them in a box."
    },
    drop_nulls: {
        label: 'Drop Missing Values',
        category: 'preprocessing',
        defaultParams: { axis: 0 },
        description: "Removes rows or columns that have 'NaN' or empty values, so the model doesn't crash."
    },
    train_test_split: {
        label: 'Train/Test Split',
        category: 'preprocessing',
        defaultParams: { test_size: 0.2 },
        description: "Splits data into a 'Study Guide' (Train) and an 'Exam' (Test) to check if the model actually learned."
    },
    logistic_regression: {
        label: 'Logistic Regression',
        category: 'model',
        defaultParams: { C: 1.0 },
        description: "A fast, simple model that draws a line (or plane) to separate classes. Great for simple Yes/No questions."
    },
    decision_tree_classifier: {
        label: 'Decision Tree Classifier',
        category: 'model',
        defaultParams: { max_depth: 5 },
        description: "Builds a flowchart-like structure (e.g. 'If Petal > 2...') to make decisions. Good for capturing complex rules."
    },
};

const WIZARD_STEPS = [
    { id: 1, title: 'Data Source' },
    { id: 2, title: 'Preprocessing' },
    { id: 3, title: 'Train/Test Split' },
    { id: 4, title: 'Model Selection' },
    { id: 5, title: 'Results' },
];

function ResultExplanation({ accuracy }: { accuracy: string }) {
    const accVal = parseFloat(accuracy.replace('%', ''));
    let grade = "Needs Improvement";
    let color = "#dc3545";
    let message = "The model is struggling to find patterns. Try adding more data or using a different model.";

    if (accVal >= 90) {
        grade = "Excellent";
        color = "#28a745";
        message = `Your model correctly identifies the class ${accVal}% of the time! It has learned the patterns very well.`;
    } else if (accVal >= 70) {
        grade = "Good";
        color = "#ffc107";
        message = `Your model is doing a decent job (correct ${accVal} times out of 100). It makes some mistakes but is generally reliable.`;
    }

    return (
        <div style={{ background: 'var(--background)', padding: '1.5rem', borderRadius: '8px', borderLeft: `5px solid ${color}`, marginBottom: '1.5rem' }}>
            <h3 style={{ color: color, marginBottom: '0.5rem' }}>{grade} Model ({accuracy})</h3>
            <p style={{ lineHeight: '1.6', fontSize: '1rem' }}>{message}</p>
        </div>
    )
}

function InfoBox({ text }: { text: string }) {
    return (
        <div style={{
            fontSize: '0.85rem',
            color: '#0d47a1',
            background: '#e3f2fd',
            padding: '0.75rem',
            borderRadius: '6px',
            marginTop: '0.5rem',
            border: '1px solid #bbdefb'
        }}>
            <strong>💡 Did you know?</strong> {text}
        </div>
    );
}

export default function StepWizard() {
    const [currentStep, setCurrentStep] = useState(1);
    const [preprocessingSteps, setPreprocessingSteps] = useState<PipelineStep[]>([]);
    const [testSize, setTestSize] = useState<number>(0.2);
    const [selectedModel, setSelectedModel] = useState<PipelineStep | null>(null);
    const [isRunning, setIsRunning] = useState(false);
    const [results, setResults] = useState<PipelineResult | null>(null);
    const [datasetMetadata, setDatasetMetadata] = useState<DatasetMetadata | null>(null);
    const [uploadError, setUploadError] = useState<string | null>(null);
    const [isUploading, setIsUploading] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);

    // === Handlers ===
    const handleNext = async () => {
        if (currentStep === 4) await runPipeline();
        if (currentStep < WIZARD_STEPS.length) setCurrentStep(currentStep + 1);
    };
    const handleBack = () => { if (currentStep > 1) setCurrentStep(currentStep - 1); };

    const addPreprocessingStep = (type: PreprocessingType) => setPreprocessingSteps([...preprocessingSteps, { id: Math.random().toString(36).substr(2, 9), type, params: { ...STEP_TEMPLATES[type].defaultParams } }]);
    const removePreprocessingStep = (id: string) => setPreprocessingSteps(preprocessingSteps.filter((s) => s.id !== id));
    const updatePreprocessingParam = (id: string, param: string, value: any) => setPreprocessingSteps(preprocessingSteps.map((s) => s.id === id ? { ...s, params: { ...s.params, [param]: value } } : s));
    const selectModel = (type: ModelType) => setSelectedModel({ id: 'model-step', type, params: { ...STEP_TEMPLATES[type].defaultParams } });
    const updateModelParam = (param: string, value: any) => selectedModel && setSelectedModel({ ...selectedModel, params: { ...selectedModel.params, [param]: value } });

    const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;
        setIsUploading(true);
        setUploadError(null);
        const formData = new FormData();
        formData.append('file', file);
        try {
            const res = await fetch(`${API_BASE}/api/upload`, { method: 'POST', body: formData });
            if (!res.ok) throw new Error((await res.json()).detail || 'Upload failed');
            setDatasetMetadata(await res.json());
        } catch (err: any) { setUploadError(err.message); setDatasetMetadata(null); } finally { setIsUploading(false); }
    };

    const runPipeline = async () => {
        if (!selectedModel) return;
        setIsRunning(true);
        setResults(null);
        const fullSteps: PipelineStep[] = [...preprocessingSteps, { id: 'split', type: 'train_test_split', params: { test_size: testSize } }, selectedModel];
        try {
            const res = await fetch(`${API_BASE}/api/pipeline/run`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ datasetName: datasetMetadata?.filename || 'Unknown', steps: fullSteps }),
            });
            if (!res.ok) throw new Error(`Server status: ${res.status}`);
            setResults(await res.json());
        } catch (error: any) {
            setResults({ success: false, message: 'Failed to execute pipeline', metrics: {}, logs: [`Error: ${error.message}`] });
        } finally { setIsRunning(false); }
    };

    const downloadModel = () => { if (results?.model_path) window.open(`${API_BASE}/api/download/${results.model_path}`, '_blank'); };

    return (
        <div className="wizard-container">
            <div className="step-indicator">
                {WIZARD_STEPS.map((step) => (
                    <div key={step.id} className={`step-item ${currentStep === step.id ? 'active' : ''} ${currentStep > step.id ? 'completed' : ''}`}>
                        <span className="step-badge">{step.id}</span>
                        <span>{step.title}</span>
                    </div>
                ))}
            </div>

            <div className="step-content">
                {/* STEP 1: DATA */}
                {currentStep === 1 && (
                    <div>
                        <h2>Data Source</h2>
                        <p className="mb-4">Upload a classification dataset (like Iris or Wine). We will try to predict the last column.</p>
                        <input type="file" ref={fileInputRef} onChange={handleFileChange} style={{ display: 'none' }} accept=".csv, .xlsx, .xls" />

                        {!datasetMetadata ? (
                            <div style={{ border: '2px dashed var(--border)', padding: '3rem', textAlign: 'center', marginTop: '1rem', cursor: 'pointer', background: isUploading ? 'rgba(0,0,0,0.05)' : 'transparent' }} onClick={() => !isUploading && fileInputRef.current?.click()}>
                                {isUploading ? <div><div className="spinner">⏳</div><p>Uploading and analyzing...</p></div> : <div><p style={{ fontSize: '1.2rem' }}>Drag and drop CSV/Excel here</p></div>}
                            </div>
                        ) : (
                            <div className="pipeline-step-card" style={{ marginTop: '1rem' }}>
                                <div className="step-card-header">
                                    <div className="step-title"><span style={{ fontSize: '1.5rem' }}>📄</span>{datasetMetadata.filename}</div>
                                    <button className="btn btn-secondary" style={{ fontSize: '0.8rem' }} onClick={() => { setDatasetMetadata(null); setPreprocessingSteps([]); setSelectedModel(null); }}>Change</button>
                                </div>

                                <div className="results-grid" style={{ marginBottom: '1rem', marginTop: '1rem' }}>
                                    <div className="metric-card"><div className="metric-value">{datasetMetadata.rows}</div><div className="metric-label">Rows</div></div>
                                    <div className="metric-card"><div className="metric-value">{datasetMetadata.columns}</div><div className="metric-label">Columns</div></div>
                                </div>

                                {/* Data Preview Table */}
                                {datasetMetadata.preview && datasetMetadata.preview.length > 0 && (
                                    <div style={{ overflowX: 'auto', marginBottom: '1rem', border: '1px solid var(--border)', borderRadius: '8px' }}>
                                        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem', textAlign: 'left' }}>
                                            <thead style={{ background: '#f8f9fa' }}>
                                                <tr>
                                                    {Object.keys(datasetMetadata.preview[0]).map((col) => (
                                                        <th key={col} style={{ padding: '0.75rem', borderBottom: '1px solid var(--border)', color: '#495057' }}>{col}</th>
                                                    ))}
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {datasetMetadata.preview.map((row, idx) => (
                                                    <tr key={idx} style={{ borderBottom: idx === datasetMetadata.preview.length - 1 ? 'none' : '1px solid #eee' }}>
                                                        {Object.values(row).map((val: any, i) => (
                                                            <td key={i} style={{ padding: '0.75rem' }}>{val?.toString() ?? ''}</td>
                                                        ))}
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>
                                )}

                                {/* Visualizations */}
                                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
                                    {/* Target Distribution */}
                                    {datasetMetadata.distributions && (
                                        <div style={{ border: '1px solid var(--border)', borderRadius: '8px', padding: '1rem' }}>
                                            <h4 style={{ marginBottom: '0.5rem', fontSize: '0.9rem', color: 'var(--secondary)' }}>Target Distribution ({datasetMetadata.distributions.target_column})</h4>
                                            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                                                {Object.entries(datasetMetadata.distributions.data).map(([label, count]) => {
                                                    const max = Math.max(...Object.values(datasetMetadata.distributions!.data));
                                                    return (
                                                        <div key={label} style={{ display: 'flex', alignItems: 'center', fontSize: '0.75rem' }}>
                                                            <div style={{ width: '60px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={label}>{label}</div>
                                                            <div style={{ flex: 1, height: '12px', background: '#f1f3f5', borderRadius: '2px', marginLeft: '8px' }}>
                                                                <div style={{ width: `${(count / max) * 100}%`, height: '100%', background: '#6f42c1', borderRadius: '2px' }}></div>
                                                            </div>
                                                            <div style={{ width: '30px', textAlign: 'right' }}>{count}</div>
                                                        </div>
                                                    )
                                                })}
                                            </div>
                                            <div style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: '#666', fontStyle: 'italic' }}>
                                                Check if bars are equal. Unbalanced bars means the model might be biased.
                                            </div>
                                        </div>
                                    )}

                                    {/* Correlation Matrix */}
                                    {datasetMetadata.correlations && (
                                        <div style={{ border: '1px solid var(--border)', borderRadius: '8px', padding: '1rem' }}>
                                            <h4 style={{ marginBottom: '0.5rem', fontSize: '0.9rem', color: 'var(--secondary)' }}>Correlation Heatmap</h4>
                                            {datasetMetadata.correlations.labels.length > 8 ? (
                                                <p style={{ fontSize: '0.8rem', color: 'var(--secondary)' }}>Too many features.</p>
                                            ) : (
                                                <div style={{ display: 'grid', gridTemplateColumns: `repeat(${datasetMetadata.correlations.labels.length}, 1fr)`, gap: '2px' }}>
                                                    {datasetMetadata.correlations.data.flat().map((val, idx) => (
                                                        <div key={idx} style={{ aspectRatio: '1', background: `rgba(111, 66, 193, ${Math.abs(val)})`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.6rem', color: Math.abs(val) > 0.5 ? 'white' : 'black', borderRadius: '2px' }}>
                                                            {val.toFixed(1)}
                                                        </div>
                                                    ))}
                                                </div>
                                            )}
                                            <div style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: '#666', fontStyle: 'italic' }}>
                                                Darker squares = Stronger relationship. If two features are always the same, you might not need both.
                                            </div>
                                        </div>
                                    )}
                                </div>

                                {/* Scatter Plot */}
                                {datasetMetadata.scatter_data && datasetMetadata.scatter_data.length > 0 && (
                                    <div style={{ border: '1px solid var(--border)', borderRadius: '8px', padding: '1rem' }}>
                                        <h4 style={{ marginBottom: '0.5rem', fontSize: '0.9rem', color: 'var(--secondary)' }}>Feature Scatter (First 2 Numeric Cols)</h4>
                                        <div style={{ height: '200px', position: 'relative', borderLeft: '1px solid #ddd', borderBottom: '1px solid #ddd' }}>
                                            {(() => {
                                                const xVals = datasetMetadata.scatter_data!.map(p => p.x);
                                                const yVals = datasetMetadata.scatter_data!.map(p => p.y);
                                                const minX = Math.min(...xVals), maxX = Math.max(...xVals);
                                                const minY = Math.min(...yVals), maxY = Math.max(...yVals);
                                                const getColor = (str: string) => {
                                                    let hash = 0;
                                                    for (let i = 0; i < str.length; i++) hash = str.charCodeAt(i) + ((hash << 5) - hash);
                                                    const c = (hash & 0x00FFFFFF).toString(16).toUpperCase();
                                                    return '#' + '00000'.substring(0, 6 - c.length) + c;
                                                }
                                                return datasetMetadata.scatter_data!.map((p, i) => (
                                                    <div key={i} style={{ position: 'absolute', left: `${((p.x - minX) / (maxX - minX)) * 95}%`, bottom: `${((p.y - minY) / (maxY - minY)) * 95}%`, width: '8px', height: '8px', background: getColor(p.class), borderRadius: '50%', opacity: 0.7 }} title={`Class: ${p.class}`}></div>
                                                ))
                                            })()}
                                        </div>
                                        <div style={{ marginTop: '0.5rem', fontSize: '0.75rem', color: '#666', fontStyle: 'italic' }}>
                                            Shows how your data is grouped. Distinct colorful clusters mean the model will learn easily.
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}
                        {uploadError && <div style={{ color: 'red', marginTop: '1rem' }}>{uploadError}</div>}
                    </div>
                )}

                {/* STEP 2: PREPROCESSING */}
                {currentStep === 2 && (
                    <div>
                        <h2>Preprocessing</h2>
                        <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1rem', flexWrap: 'wrap' }}>
                            <button className="add-btn" onClick={() => addPreprocessingStep('standard_scaler')}>+ Standard Scaler</button>
                            <button className="add-btn" onClick={() => addPreprocessingStep('min_max_scaler')}>+ Min Max</button>
                            <button className="add-btn" onClick={() => addPreprocessingStep('drop_nulls')}>+ Drop Nulls</button>
                        </div>
                        {preprocessingSteps.length > 0 && <p className="mb-4" style={{ fontSize: '0.9rem', color: 'var(--secondary)' }}>Added steps:</p>}
                        {preprocessingSteps.map((step, index) => (
                            <div key={step.id} className="pipeline-step-card">
                                <div className="step-card-header">
                                    <div><span style={{ color: 'var(--secondary)', fontSize: '0.8rem' }}>#{index + 1}</span> {STEP_TEMPLATES[step.type].label}</div>
                                    <button className="remove-btn" onClick={() => removePreprocessingStep(step.id)}>×</button>
                                </div>
                                <InfoBox text={STEP_TEMPLATES[step.type].description} />
                            </div>
                        ))}
                    </div>
                )}

                {/* STEP 3: SPLIT */}
                {currentStep === 3 && (
                    <div>
                        <h2>Train / Test Split</h2>
                        <div className="pipeline-step-card" style={{ padding: '2rem' }}>
                            <div style={{ marginBottom: '2rem' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                                    <span>Training Data: <strong>{Math.round((1 - testSize) * 100)}%</strong> (Study)</span>
                                    <span>Testing Data: <strong>{Math.round(testSize * 100)}%</strong> (Exam)</span>
                                </div>
                                <input type="range" min="0.1" max="0.9" step="0.05" value={testSize} onChange={(e) => setTestSize(parseFloat(e.target.value))} style={{ width: '100%' }} />
                            </div>
                            <div style={{ display: 'flex', gap: '1rem' }}>
                                <div style={{ flex: 1 - testSize, background: '#e3f2fd', height: '30px', borderRadius: '4px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.8rem', color: '#0d47a1' }}>Train</div>
                                <div style={{ flex: testSize, background: '#fce4ec', height: '30px', borderRadius: '4px', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.8rem', color: '#880e4f' }}>Test</div>
                            </div>
                            <InfoBox text={STEP_TEMPLATES['train_test_split'].description} />
                        </div>
                    </div>
                )}

                {/* STEP 4: MODEL */}
                {currentStep === 4 && (
                    <div>
                        <h2>Model Selection</h2>
                        <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem' }}>
                            {['logistic_regression', 'decision_tree_classifier'].map((modelType) => (
                                <div key={modelType} onClick={() => selectModel(modelType as ModelType)} style={{ flex: 1, padding: '1.5rem', border: `2px solid ${selectedModel?.type === modelType ? 'var(--primary)' : 'var(--border)'}`, borderRadius: '8px', cursor: 'pointer', background: selectedModel?.type === modelType ? 'rgba(0,0,0,0.02)' : 'transparent' }}>
                                    <h3 style={{ marginBottom: '0.5rem' }}>{STEP_TEMPLATES[modelType as StepType].label}</h3>
                                    <p style={{ fontSize: '0.8rem', color: 'var(--secondary)', lineHeight: '1.4' }}>{STEP_TEMPLATES[modelType as StepType].description}</p>
                                </div>
                            ))}
                        </div>
                        {selectedModel && <div className="pipeline-step-card"><p>Selected: <strong>{STEP_TEMPLATES[selectedModel.type].label}</strong></p></div>}
                    </div>
                )}

                {/* STEP 5: RESULTS */}
                {currentStep === 5 && (
                    <div>
                        <h2>Results</h2>
                        {isRunning ? <div style={{ textAlign: 'center', padding: '3rem' }}><div className="spinner">⏳</div><p>Training...</p></div> : results && (
                            <>
                                {results.success ? (
                                    <>
                                        {results.metrics.Accuracy && <ResultExplanation accuracy={results.metrics.Accuracy} />}

                                        <div className="results-grid">
                                            <div className="metric-card" style={{ color: '#28a745', borderColor: '#28a745' }}><div className="metric-value">{results.metrics.Accuracy}</div><div className="metric-label">ACCURACY</div></div>
                                            <div className="metric-card"><div className="metric-value">{results.metrics.train_samples}</div><div className="metric-label">TRAIN SAMPLES</div></div>
                                            <div className="metric-card"><div className="metric-value">{results.metrics.test_samples}</div><div className="metric-label">TEST SAMPLES</div></div>
                                        </div>


                                        {/* Feature Importance */}
                                        {results.metrics.feature_importance && (
                                            <div style={{ marginBottom: '1.5rem', border: '1px solid var(--border)', borderRadius: '8px', padding: '1.5rem' }}>
                                                <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem' }}>Feature Importance (How logic works)</h3>
                                                <p style={{ marginBottom: '1rem', fontSize: '0.9rem', color: '#666' }}>The longer the bar, the more the model relies on this feature.</p>
                                                <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                                                    {results.metrics.feature_importance.slice(0, 10).map((feat: any, idx: number) => {
                                                        const maxVal = Math.max(...results.metrics.feature_importance.map((f: any) => f.importance));
                                                        return (
                                                            <div key={idx} style={{ display: 'flex', alignItems: 'center', fontSize: '0.85rem' }}>
                                                                <div style={{ width: '120px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={feat.feature}>{feat.feature}</div>
                                                                <div style={{ flex: 1, background: '#f1f3f5', height: '18px', borderRadius: '4px', overflow: 'hidden', margin: '0 10px' }}>
                                                                    <div style={{ width: `${(feat.importance / maxVal) * 100}%`, height: '100%', background: '#007bff' }}></div>
                                                                </div>
                                                                <div style={{ width: '60px', textAlign: 'right', fontSize: '0.75rem' }}>{feat.importance.toFixed(3)}</div>
                                                            </div>
                                                        )
                                                    })}
                                                </div>
                                            </div>
                                        )}

                                        {/* Confusion Matrix */}
                                        {results.metrics.confusion_matrix && (
                                            <div style={{ marginBottom: '1.5rem', border: '1px solid var(--border)', borderRadius: '8px', padding: '1.5rem' }}>
                                                <h3 style={{ marginBottom: '1rem', fontSize: '1.1rem' }}>Confusion Matrix (Where it fails)</h3>
                                                <p style={{ marginBottom: '1rem', fontSize: '0.9rem', color: '#666' }}>Rows = Actual, Columns = Predicted. <br /><strong>Diagonal (Green)</strong> = Correct. <strong>Off-Diagonal</strong> = Mistakes.</p>
                                                <div style={{ display: 'grid', width: 'fit-content', gap: '1px', background: '#dee2e6', border: '1px solid #dee2e6' }}>
                                                    {results.metrics.confusion_matrix.map((row: number[], rIdx: number) => (
                                                        <div key={rIdx} style={{ display: 'flex', gap: '1px' }}>
                                                            {row.map((val: number, cIdx: number) => (
                                                                <div key={cIdx} style={{ width: '50px', height: '50px', background: rIdx === cIdx ? '#d4edda' : '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.9rem', color: rIdx === cIdx ? '#155724' : '#495057' }}>
                                                                    {val}
                                                                </div>
                                                            ))}
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        )}

                                        {results.model_path && <button className="btn btn-primary" onClick={downloadModel} style={{ width: '100%', marginBottom: '1rem' }}>📥 Download Trained Model</button>}
                                    </>
                                ) : (
                                    <div style={{ padding: '1rem', background: '#f8d7da', color: '#721c24', borderRadius: '4px' }}>{results.message}</div>
                                )}
                                <div className="logs-console">{results.logs.map((log, i) => <div key={i}>{log}</div>)}</div>
                            </>
                        )}
                    </div>
                )}

            </div>
            <div className="wizard-controls">
                <button className="btn btn-secondary" onClick={handleBack} disabled={currentStep === 1 || isRunning}>Back</button>
                {currentStep < 5 && <button className="btn btn-primary" onClick={handleNext} disabled={isRunning || (currentStep === 1 && !datasetMetadata) || (currentStep === 4 && !selectedModel)}>{currentStep === 4 ? 'Run Training' : 'Next'}</button>}
                {currentStep === 5 && <button className="btn btn-primary" onClick={() => setCurrentStep(1)}>Start Over</button>}
            </div>
        </div >
    );
}
