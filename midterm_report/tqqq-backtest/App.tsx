import React from 'react';
import Dashboard from './components/Dashboard';

const App: React.FC = () => {
  return (
    <div className="bg-gray-900 text-gray-200 min-h-screen font-sans">
      <div className="container mx-auto">
        <Dashboard />
      </div>
       <footer className="text-center py-4 mt-8 border-t border-gray-700">
        <p className="text-sm text-gray-500">
          This is a simulation tool for educational purposes only. Past performance is not indicative of future results. Not financial advice.
        </p>
      </footer>
    </div>
  );
};

export default App;
