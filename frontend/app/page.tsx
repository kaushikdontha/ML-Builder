import StepWizard from '../components/StepWizard';

export default function Home() {
  return (
    <main style={{ minHeight: '100vh', padding: '2rem', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <h1 style={{ marginBottom: '2rem', fontSize: '2.5rem' }}>ML Builder</h1>
      <StepWizard />
    </main>
  );
}
