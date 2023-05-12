import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import PageContainer from './components/PageContainer';
import Home from './components/pages/Home';
import Videos from './components/pages/Videos';
import Blog from './components/pages/Blog';
import Resources from './components/pages/Resources';
import Contact from './components/pages/Contact';

const AppRouter = () => {
    return (
      <Router>
        <Routes>
            <Route path="/" element={
                <PageContainer Page={Home} />} 
            />
            
            <Route path="/videos" element={
                <PageContainer Page={Videos} />} 
            />

            <Route path="/blog"element={
                <PageContainer Page={Blog} />}
            />

            <Route path="/resources" element={
                <PageContainer Page={Resources} />}
            />

            <Route path="/contact" element={
                <PageContainer Page={Contact} />}
            />

            <Route path="/blog/:blogId" element={
                <PageContainer Page={Blog} />} 
            />

        </Routes>
      </Router>
    );
  }
  
  export default AppRouter;
